### Phase 1: Frontend UI Generation

*   **Tool Used:** Google Gemini 3 Pro
*   **Exact Prompt:**
    > Act as a Frontend Engineer. I need a single-file `home.html` interface for a Computer Vision API demo.
    >
    > **Visual Style:**
    > - "Industrial/Logistics" theme: Dark mode, Slate-900 backgrounds, monospace fonts for data, and high-contrast green/blue accents for status.
    > - Use vanilla CSS (no Tailwind/Bootstrap CDN) to keep it self-contained.
    >
    > **Functional Requirements:**
    > 1. **Upload Zone:** A large, centered drag-and-drop area for video files (MP4/AVI).
    > 2. **Preview:** When a file is dropped, show a `<video>` preview player immediately.
    > 3. **Action:** A "Analyze Temporal Sequence" button that sends the file via `POST` to `http://localhost:8000/predict`.
    > 4. **Response Handling:** The API returns JSON with keys: `{ "dominant_operation": str, "confidence": float, "anticipated_next_operation": str }`.
    >
    > **Output Display:**
    > - Parse the JSON and display "Current Operation" and "Next Anticipated Action" in large, distinct metric cards.
    > - Display the raw JSON response in a styled code block for debugging.
    > - Include a loading spinner state while waiting for the inference.

*   **Code Accepted vs. Modified:**
    *   **Accepted:** The CSS Grid layout for the metric cards and the specific color palette (`#0f172a` background) were used as-is. The drag-and-drop event listeners (`dragover`, `drop`) worked perfectly out of the box.
    *   **Modified:** The AI initially tried to send the file as a JSON string. I manually rewrote the JavaScript `fetch` logic to use a `FormData` object so it correctly interfaces with FastAPI's `UploadFile` type. I also added a check to hide the "Results" div when a new video is selected to prevent stale data from showing.

*   **Estimated Time Saved:** 50 minutes (Saved writing 150+ lines of CSS/JS boilerplate and styling).

### Phase 2: FastAPI Video Inference Endpoint

*   **Tool Used:** Google Gemini 3 Pro
*   **Exact Prompt:**
    > Write a production-ready FastAPI endpoint `@app.post("/predict")` for a Video Language Model (Qwen2-VL).
    >
    > **Requirements:**
    > 1. **Input:** Accept a video file via `UploadFile`.
    > 2. **File Handling:** Since the video processing library (`decord`) requires a file path, safely save the upload to a named temporary file, process it, and ensure cleanup (deletion) in a `finally` block.
    > 3. **Model Interaction:** Use the global `model` and `processor` (assume they are already loaded). Construct a prompt asking for: "dominant_operation", "temporal_segment" (start/end frames), and "anticipated_next_operation".
    > 4. **Vision Utils:** Use `qwen_vl_utils.process_vision_info` to handle the video inputs. Set `fps=1.0` to conserve VRAM.
    > 5. **Output Parsing:** The model returns raw text. Parse this into a clean JSON object. Handle the case where the model wraps the JSON in markdown code blocks.
    >
    > **Return Schema:**
    > ```json
    > {
    >   "clip_id": "filename",
    >   "dominant_operation": "String",
    >   "temporal_segment": { "start_frame": int, "end_frame": int },
    >   "anticipated_next_operation": "String",
    >   "confidence": float
    > }
    > ```

*   **Code Accepted vs. Modified:**
    *   **Accepted:** The `tempfile.NamedTemporaryFile` logic with `shutil.copyfileobj` was adopted as-is. The `process_vision_info` implementation saved significant time looking up documentation.
    *   **Modified:** The AI initially placed the `model.generate()` call inside a generic try/except block that swallowed errors. I refactored the error handling to return specific HTTP 500 errors with the model's raw output if JSON parsing fails. I also added a custom `clean_json_output` utility function because the base model kept outputting "```json" markdown tags which caused `json.loads` to crash.

*   **Estimated Time Saved:** 45 minutes (Avoided debugging complex `decord` video loading syntax and `transformers` input formatting).
