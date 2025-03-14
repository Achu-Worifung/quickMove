import speech_recognition as sr
import wave
import time 


r = sr.Recognizer()

# Counter for naming the audio files uniquely
file_counter = 1

with sr.Microphone() as source:
    
    print("Say something! (Say 'exit' to stop)")

    while True:
        audio = r.listen(source)  # Listen for speech
        

        try:
            start_time = time.perf_counter()
            text = r.recognize_google(audio)  # Convert speech to text
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            print("You said:", text)

            # Exit condition
            if text.lower() == "exit":
                print("Exiting program...")
                break

            # Save the audio file
            filename = f"recording_{file_counter}.wav"
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())

            print(f"Saved recording as {filename}")
            file_counter += 1  # Increment file counter

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
