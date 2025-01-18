"""
generate_podcast_example.py

End-to-end example demonstrating how to create an AI-generated podcast from a text script 
using Podcastfy, with Two Speakers (Person1, Person2).

Steps:
1) Ensure you have a .env file with your ELEVENLABS_API_KEY (or other TTS model keys).
2) pip install podcastfy (and 'pip install ffmpeg' if needed).
3) Replace the 'expanded_script_text' with your real text that has <Person1> ... </Person1> 
   and <Person2> ... </Person2> tags.
4) Run:  python generate_podcast_example.py
"""

import os
from podcastfy.client import generate_podcast

# Optional: If you'd like the audio to be embedded in a Jupyter Notebook, you'll do:
# from IPython.display import Audio, display

# 1) Provide your final script with <Person1> and <Person2> tags
expanded_script_text = """
<Person1>Hey there, cosmic explorers! Welcome back to our show, 
all about hunting tiny primordial black holes in the Solar System...</Person1>

<Person2>That’s right, I'm your co-host, and together we’ll dig into ephemerides...</Person2>

<Person1>First up, let's talk about dark matter. 
Could these little black holes make up that mysterious stuff?...</Person1>

<Person2>We'll also see how NASA’s Planetary Data System might contain the clues we need...</Person2>

... 
<Person1>Don't forget to share this episode and keep exploring the universe!</Person1>
<Person2>See you next time, folks!</Person2>
""".strip()

def main():
    """
    Main function to generate an AI-voiced podcast from a 2-person script.
    """
    # 2) Call generate_podcast, passing 'text=expanded_script_text'
    #    and specifying TTS model. Here we use 'elevenlabs' as an example.
    #    For other TTS providers, pass tts_model='openai', 'edge', 'gemini', or 'geminimulti', etc.
    try:
        audio_file_path = generate_podcast(
            text=expanded_script_text,
            tts_model="elevenlabs",  # or "openai"/"gemini"/"edge"/"geminimulti"
            # conversation_config={}  # Optionally pass a dictionary with custom settings
        )
        print(f"✅ Podcast audio generated: {audio_file_path}")

        # 3) (Optional) If you were in a Jupyter notebook, you could embed the result:
        # display(Audio(audio_file_path))
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
