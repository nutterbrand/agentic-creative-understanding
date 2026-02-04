# AI-Powered Creative Performance Analyzer

[cite_start]This project is a prototype demonstrating how Generative AI (Google Gemini) can be used to analyze creative assets—static images or video frames—to determine **why** certain creatives perform better than others[cite: 1].

[cite_start]By correlating performance data (CTR, CVR, etc.) with visual elements, this tool automatically generates an "annotated" audit of your creative strategy, drawing attention to specific components driving user behavior[cite: 1].

## 🎯 Project Goal

The goal is to move beyond simple "winner/loser" metrics and gain a **visual point of view** on creative performance. This tool answers questions like:
* *"Why did Creative A get a higher Click-Through Rate than Creative B?"*
* [cite_start]*"Does showing the product packaging drive more trust (and conversions) than showing the product in use?"* [cite: 15, 20]

## 🧠 How It Works

[cite_start]The script `creative_gemini_visual_annotator.py` automates the following workflow[cite: 1]:

1.  **Ingestion:** It consumes a folder of media files (images/videos) and a CSV of performance metrics (e.g., Impressions, Clicks, Conversions).
2.  **Pairwise Comparison:** It identifies top-performing and bottom-performing creatives for specific metrics (e.g., High CTR vs. Low CTR).
3.  **Visual AI Analysis:** It sends both images to Google's **Gemini** model with a specific prompt: *"Compare these two images. Identify visible differences that plausibly explain why A performed better than B."*
4.  **Structured Annotation:** Gemini returns structured JSON data containing bounding box coordinates and explanations for key elements (e.g., "Prominent Amazon Badge," "Human Hands").
5.  **Rendering:** The script uses `Pillow` to physically draw colored circles and labels onto the images based on the AI's coordinates:
    * 🟢 **Green:** Elements associated with higher performance.
    * 🔴 **Red:** Elements associated with lower performance.
6.  **Reporting:** Finally, it compiles the annotated images, the AI's reasoning, and the performance data into a formatted **Word Document (.docx)** for easy sharing with design teams.

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* Gemini API Key.
* `ffmpeg` (optional, for processing video files).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/creative-analyzer.git](https://github.com/yourusername/creative-analyzer.git)
    cd creative-analyzer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Authentication:**
    * **Option A (Vertex AI - Recommended):** Ensure you have the Google Cloud CLI installed and authenticated (`gcloud auth login`).
    * **Option B (API Key):** Set your environment variable:
        ```bash
        export GEMINI_API_KEY="your-api-key-here"
        ```

### Usage

[cite_start]Run the script by pointing it to your media folder and your metrics CSV[cite: 1]:

```bash
python creative_gemini_visual_annotator.py \
  --media-dir ./assets \
  --metrics-csv ./metrics.csv \
  --out-dir ./output_report \
  --metrics click_through_rate conversion_rate \
  --pairwise

### Sample Output
[annotated_creative_report.docx](https://github.com/user-attachments/files/25078444/annotated_creative_report.docx)

