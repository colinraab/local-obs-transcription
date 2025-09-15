import ffmpeg
import whisper
import sys
import os
import tempfile
import shutil
import argparse
# New imports for Hugging Face transformers
import torch
from transformers import pipeline

def summarize_text(transcription_text, model_name="google/gemma-3-270m-it"):
    """
    Summarizes the given text using a Hugging Face transformers pipeline.

    Args:
        transcription_text (str): The text to summarize.
        model_name (str): The name of the model on the Hugging Face Hub.

    Returns:
        str: The generated summary.
    """
    print(f"\nLoading summarization model '{model_name}' from Hugging Face...")
    try:
        # Use a pipeline for high-level, easy-to-use text generation.
        # device_map="auto" will use a GPU if available (CUDA or MPS on Mac).
        summarizer = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Summarization model loaded.")

        # Use a chat template for robust and model-specific prompting
        messages = [
            {
                "role": "user",
                "content": f"You are an expert summarizer. Summarize the following conversation into a concise paragraph. Capture the key topics and conclusions.\n\nConversation:\n{transcription_text}"
            },
        ]
        
        prompt = summarizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print("Generating summary...")
        outputs = summarizer(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        
        # The output contains the full prompt and the generated text.
        # We need to extract just the generated part.
        summary = outputs[0]["generated_text"][len(prompt):].strip()
        print("Summary generated.")
        return summary

    except ImportError:
        print("\nError: `transformers`, `torch`, or `accelerate` not installed.", file=sys.stderr)
        print("Please run: pip install transformers torch accelerate", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred during summarization: {e}", file=sys.stderr)
        print("This may be due to a missing Hugging Face authentication token for a gated model.", file=sys.stderr)
        print("Please run `huggingface-cli login` in your terminal and ensure you have accepted the model's terms.", file=sys.stderr)
        return None


def transcribe_multitrack_audio(input_file_path, do_summarization, model_name_for_summary):
    """
    Separates a multi-track audio file, transcribes each track using Whisper,
    merges the transcriptions, and optionally generates a summary.

    Args:
        input_file_path (str): The path to the input Opus audio file.
        do_summarization (bool): Whether to generate a summary.
        model_name_for_summary (str): The Hugging Face model to use for summarization.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    print("Loading Whisper model...")
    # You can choose other model sizes like "tiny", "base", "small", "medium", "large"
    model = whisper.load_model("base")
    print("Model loaded.")

    base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    output_transcription_path = f"{base_filename}_transcription.txt"
    full_transcription_content = []

    # Create a temporary directory to store the separated audio tracks
    with tempfile.TemporaryDirectory() as temp_dir:
        track1_path = os.path.join(temp_dir, "track1.wav")
        track2_path = os.path.join(temp_dir, "track2.wav")

        try:
            print("Separating audio tracks...")
            # Separate Track 1 (Desktop Audio)
            ffmpeg.input(input_file_path).output(track1_path, map='0:a:0').run(overwrite_output=True, quiet=True)
            # Separate Track 2 (Microphone)
            ffmpeg.input(input_file_path).output(track2_path, map='0:a:1').run(overwrite_output=True, quiet=True)
            print("Tracks separated successfully.")

            print("Transcribing Desktop Audio (Track 1)...")
            result_track1 = model.transcribe(track1_path)
            print("Desktop Audio transcribed.")

            print("Transcribing Microphone (Track 2)...")
            result_track2 = model.transcribe(track2_path)
            print("Microphone transcribed.")

            # Add labels to the segments
            for segment in result_track1['segments']:
                segment['source'] = 'Desktop Audio'
            for segment in result_track2['segments']:
                segment['source'] = 'Microphone'

            # Merge and sort segments by start time
            all_segments = sorted(
                result_track1['segments'] + result_track2['segments'],
                key=lambda x: x['start']
            )

            print(f"Merging transcriptions into '{output_transcription_path}'...")
            with open(output_transcription_path, 'w', encoding='utf-8') as f:
                for segment in all_segments:
                    start_time = int(segment['start'])
                    end_time = int(segment['end'])
                    
                    start_h, rem = divmod(start_time, 3600)
                    start_m, start_s = divmod(rem, 60)
                    end_h, rem = divmod(end_time, 3600)
                    end_m, end_s = divmod(rem, 60)

                    timestamp = f"[{start_h:02}:{start_m:02}:{start_s:02} -> {end_h:02}:{end_m:02}:{end_s:02}]"
                    source = segment['source']
                    text = segment['text'].strip()
                    
                    line = f"{timestamp} {source}: {text}\n"
                    f.write(line)
                    full_transcription_content.append(text) # Just append text for a cleaner summary
            
            print("Transcription complete.")

            # Generate and append summary
            if do_summarization:
                transcription_for_summary = "\n".join(full_transcription_content)
                summary = summarize_text(transcription_for_summary, model_name=model_name_for_summary)
                if summary:
                    with open(output_transcription_path, 'a', encoding='utf-8') as f:
                        f.write("\n\n====================\n")
                        f.write("       SUMMARY\n")
                        f.write("====================\n\n")
                        f.write(summary)
                    print(f"Summary appended to '{output_transcription_path}'.")

        except ffmpeg.Error as e:
            print("An error occurred with ffmpeg:", file=sys.stderr)
            print(e.stderr.decode(), file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a multi-track audio file and optionally summarize it.")
    parser.add_argument("audio_file", help="Path to the input audio file.")
    parser.add_argument("--no-summary", action="store_true", help="Disable the summarization step.")
    parser.add_argument("--model", default="google/gemma-3-270m-it", help="The Hugging Face model to use for summarization. Defaults to 'google/gemma-3-270m-it'.")
    args = parser.parse_args()
    
    # Check if ffmpeg is installed
    if not shutil.which("ffmpeg"):
         print("Error: ffmpeg is not installed or not in your PATH.", file=sys.stderr)
         print("Please install ffmpeg to run this script.", file=sys.stderr)
         print("On macOS, you can use Homebrew: 'brew install ffmpeg'", file=sys.stderr)
         sys.exit(1)

    transcribe_multitrack_audio(args.audio_file, not args.no_summary, args.model)

