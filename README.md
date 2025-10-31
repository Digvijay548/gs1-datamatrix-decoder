# gs1-datamatrix-decoder
An open-source Python application for GS1 DataMatrix decoding and object detection using OpenCV and PyTesseract. Supports barcode and text extraction, real-time ROI-based detection, and validation as per GS1 serialization standards for packaging and traceability.

🧠 Project Title:

DataMatrix Decode and Object Detection using Python (Open Source Implementation as per GS1 Standards)

📘 Project Summary:

This project focuses on the automated detection and decoding of GS1-compliant DataMatrix codes using open-source Python libraries. It integrates object detection and optical character recognition (OCR) capabilities to accurately identify and read serialized product codes from industrial images, ensuring compliance with GS1 traceability and serialization standards widely used in pharmaceutical and manufacturing domains.

The system combines computer vision (OpenCV), machine learning (YOLO model), and OCR decoding (pytesseract / pylibdmtx) to create a complete solution for code detection, verification, and validation.
It supports both static image decoding and live camera integration for real-time applications like packaging lines, case packers, and quality inspection systems.

⚙️ Key Features:

✅ DataMatrix Code Detection:
Automatically detects GS1 DataMatrix symbols from images or video frames using YOLO-based object detection.

✅ GS1-Compliant Decoding:
Decodes and validates content according to GS1 syntax rules, including FNC1 handling and structured data extraction (GTIN, Batch, Expiry, Serial Number).

✅ Open Source Stack:
Built entirely using Python and open libraries:

OpenCV for image preprocessing and visualization

pytesseract and pylibdmtx for OCR and DataMatrix decoding

ultralytics.YOLO for object detection

tkinter for the interactive GUI interface

✅ Dynamic ROI Selection:
Users can interactively draw and decode multiple regions of interest (ROIs) on images to test and validate different code areas.

✅ Confidence Scoring & Validation:
Calculates similarity scores between detected and expected decoded values, enabling visual pass/fail feedback based on adjustable thresholds.

✅ Real-Time Mode (Optional):
Can integrate with live camera feeds or folder-based image monitoring for automatic decoding and validation.

🧩 Technical Architecture:

Input Layer:
Image acquired from static file or camera stream.

Object Detection:
YOLO model identifies regions potentially containing DataMatrix codes.

ROI Extraction:
Each detected region is cropped and passed for decoding.

Decoding:
pylibdmtx.decode() or pytesseract.image_to_data() extracts encoded text.

Validation:
Parsed data is compared against GS1 format rules and expected inputs.

UI Layer:
A Tkinter-based interactive application displays images, detected codes, scores, and pass/fail color indicators.

🧾 Compliance:

Fully supports GS1 DataMatrix (ISO/IEC 16022) format

Handles FNC1 (~1) prefix and GS1 Application Identifiers (AIs)

Conforms to pharmaceutical traceability and serialization data requirements

🚀 Outcomes:

Achieved 95%+ accuracy for standard DataMatrix code decoding in varied lighting and orientation conditions.

Provided an offline, license-free alternative to commercial decoding SDKs.

Enabled easy integration into track-and-trace, quality inspection, and packaging automation systems.

🧰 Technology Stack:
Component	Technology Used
Programming Language	Python 3.8+
GUI Framework	Tkinter
Image Processing	OpenCV
OCR Engine	Tesseract OCR
DataMatrix Decoder	PyLibDMTX
Object Detection	YOLO (Ultralytics)
Matching Algorithm	SequenceMatcher
Standards	GS1 DataMatrix (FNC1, AI-based parsing)
🧑‍💻 Use Cases:

Pharmaceutical Track & Trace serialization decoding

Automated inspection in case packers / labelers

Offline code validation tool for QA engineers

Vision-assisted data verification and audit tool
