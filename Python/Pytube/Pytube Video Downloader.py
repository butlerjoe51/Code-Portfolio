from pytube import YouTube #Install with Pip before running the program

def Download(link): #Download function
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")
    
link = input("Enter the YouTube video URL: ") #Copy and paste the URL, and you can have the video saved to the directory this program was run from
Download(link)
