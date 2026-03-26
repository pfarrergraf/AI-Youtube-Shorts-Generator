import os
from pytubefix import YouTube
import ffmpeg

def get_video_size(stream):

    return stream.filesize / (1024 * 1024)

def download_youtube_video(url):
    try:
        yt = YouTube(url)

        video_streams = yt.streams.filter(type="video").order_by('resolution').desc()
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Show available streams
        print("\nAvailable video streams:")
        for i, stream in enumerate(video_streams[:5]):  # Show top 5 options
            size = get_video_size(stream)
            stream_type = "Progressive" if stream.is_progressive else "Adaptive"
            print(f"  {i}. Resolution: {stream.resolution}, Size: {size:.2f} MB, Type: {stream_type}")
        
        # Auto-select highest quality
        selected_stream = video_streams[0]
        print(f"\nAuto-selected highest quality: {selected_stream.resolution}")
        
        # Confirm selection
        if selected_stream is None:
            selected_stream = video_streams[0]
        
        size = get_video_size(selected_stream)
        stream_type = "Progressive" if selected_stream.is_progressive else "Adaptive"
        print(f"\nFinal selection: {selected_stream.resolution}, Size: {size:.2f} MB, Type: {stream_type}")

        if not os.path.exists('videos'):
            os.makedirs('videos')

        # Sanitize title for use in filename (remove invalid characters)
        safe_title = yt.title.replace("|", "-").replace(":", "-").replace("?", "").replace("*", "").replace("<", "").replace(">", "").replace('"', "").replace("/", "-").replace("\\", "-")

        print(f"Downloading video: {yt.title}")
        video_file = selected_stream.download(output_path='videos', filename_prefix="video_")

        if not selected_stream.is_progressive:
            print("Downloading audio...")
            audio_file = audio_stream.download(output_path='videos', filename_prefix="audio_")

            print("Merging video and audio...")
            output_file = os.path.join('videos', f"{safe_title}.mp4")
            stream = ffmpeg.input(video_file)
            audio = ffmpeg.input(audio_file)
            stream = ffmpeg.output(stream, audio, output_file, vcodec='libx264', acodec='aac', strict='experimental')
            ffmpeg.run(stream, overwrite_output=True)

            os.remove(video_file)
            os.remove(audio_file)
        else:
            output_file = video_file

        
        print(f"Downloaded: {yt.title} to 'videos' folder")
        print(f"File path: {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have the latest version of pytube and ffmpeg-python installed.")
        print("You can update them by running:")
        print("pip install --upgrade pytube ffmpeg-python")
        print("Also, ensure that ffmpeg is installed on your system and available in your PATH.")

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    download_youtube_video(youtube_url)
