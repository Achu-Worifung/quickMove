# **QuickMove: AI-Driven Presentation Companion**

## **Overview**

**QuickMove** is a high-performance Windows desktop application designed to automate scripture retrieval and display for professional presentation environments like **ProPresenter 7** and **VideoPsalm**. By integrating real-time speech-to-text, NLP-based classification, and Computer Vision, QuickMove reduces manual search latency by over **50%**, ensuring seamless transitions during live productions.

---

## **1. System Architecture**

QuickMove is built with **PyQt5**, leveraging a modular, event-driven UI architecture designed to handle concurrent AI subsystems while maintaining zero-lag performance.

### **The Human-in-the-Loop Workflow**

The system follows a structured pipeline to combine AI speed with human oversight:

**Audio Transcription:** Captures the speaker's voice using a transformer-based **ASR (Automated Speech Recognition)** model for high-accuracy transcription.

**Contextual Classification:** A fine-tuned **DistilBERT** model analyzes the text to determine if it is a biblical reference.

**Visual Extraction:** An **OpenCV + OCR** pipeline extracts on-screen text from video feeds to synchronize slides with live content.

**User Validation:** If a reference is detected, the application surfaces the verse for the operator to review and push live.

---

## **2. Key Features & Implementation**

### **AI & NLP Integration**

**DistilBERT Classification:** Utilized for intelligent text classification to improve search relevance and contextual accuracy.

**Real-Time ASR:** Integrated a transformer-based speech-to-text system capable of handling live event environments.

**Computer Vision (OCR):** Deployed **OpenCV** pipelines to automate synchronization between video feeds and presentation slides.

### **Technical Specifications**

**Frontend:** Developed a responsive **PyQt5 GUI** with optimized multi-threading to ensure the interface remains fluid during high-stakes operations.

**Efficiency:** Streamlined the scripture display process, resulting in a **50% reduction in latency** compared to manual searching.


---

## **3. Setup and Installation**

### **Prerequisites**

**Operating System:** Windows (for ProPresenter/VideoPsalm compatibility).

  **Python Version:** Python 3.9+ recommended.

### **Installation**

1. **Clone the Repository:**
```bash
git clone https://github.com/Achu-Worifung/quickMove.git
cd QuickMove

```


2. **Create a Virtual Environment:**
```bash
python -m venv .venv
.venv\Scripts\activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```



---

## **4. Usage**

1. Launch the application: `python main.py`.
2. Select your preferred presentation software (**ProPresenter 7** or **VideoPsalm**).


3. The ASR and OCR modules will begin monitoring live feeds and automatically suggest relevant scripture as it is mentioned or displayed.

## **5. Technical Challenges & Optimizations**
**Challenge: GUI Freezing During AI Inference**
In an event-driven application like QuickMove, the main thread handles all user interactions and interface updates. Running resource-intensive tasks—such as real-time speech transcription (FasterWhisper) or OCR scanning (OpenCV)—directly on this main thread would cause the GUI to become unresponsive or "freeze" until the task completes.

**The Solution: Decoupled Worker Architecture**
To maintain a "zero-lag" experience, I implemented a decoupled architecture using PyQt's QThread and QObject system:

Threaded Workers: Each AI subsystem (ASR, BERT, and OCR) is assigned to its own dedicated worker thread using moveToThread(). This ensures that heavy computation does not block the main event loop.

Asynchronous Communication: Instead of directly modifying UI elements from background threads—which is not thread-safe and can cause crashes—I utilized PyQt Signals and Slots.

Worker Signals: Background threads emit custom pyqtSignal objects containing transcription text or verse references.

UI Slots: The main thread's slots receive these signals and update the display safely.

Thread-Safe Concurrency: Since pyqtSignals are inherently thread-safe, they act as a secure queue for transferring data across threads without the need for complex manual locking mechanisms.

**Performance Impact**
By moving long-running tasks out of the main thread, QuickMove remains fully responsive to user input (such as verse confirmation or manual overrides) while simultaneously processing live audio and video feeds in the background.
