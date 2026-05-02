# PDF & Image → Structured Excel Extractor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-0B75C9)
![OpenCV](https://img.shields.io/badge/Image%20Processing-OpenCV-5C3EE8)
![Pandas](https://img.shields.io/badge/Data-pandas-150458)
![Excel](https://img.shields.io/badge/Export-openpyxl-217346)
![CLI](https://img.shields.io/badge/CLI-Click%20%2B%20Rich-FFDD00)

Professional OCR pipeline that converts PDFs and scanned images into clean Excel workbooks and JSON exports. It is designed for invoices, forms, receipts, reports, tabular scans, and messy multi-page documents.

## ✨ What It Does

- 📄 Accepts PDFs and images: JPG, PNG, TIFF, BMP
- 🔍 Runs PaddleOCR with angle classification
- 🧼 Improves OCR accuracy with OpenCV preprocessing: grayscale, denoise, CLAHE, adaptive thresholding, deskew
- 📊 Detects tables using Y-axis row grouping and X-axis column clustering
- 🧾 Extracts key-value pairs from forms and invoices using regex plus positional heuristics
- 📚 Handles multi-page PDFs and multi-frame TIFF images
- 📤 Exports formatted `.xlsx` files and complete `.json` data
- 🚀 Supports bulk folder processing with one master Excel workbook
- 🎨 Uses Rich progress bars, previews, summaries, and clear terminal errors

## 🧰 Tech Stack

- Python 3.10+
- PaddleOCR
- pdf2image
- OpenCV
- pandas
- openpyxl
- rich
- Click
- Pillow

## 📦 Installation

Clone the project, create a virtual environment, and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PDF conversion requires Poppler because `pdf2image` uses it internally.

- Windows: install Poppler and add its `bin` folder to `PATH`
- macOS: `brew install poppler`
- Linux: `sudo apt-get install poppler-utils`

PaddleOCR downloads its English OCR models on first run, so the first extraction may take longer.

## ▶️ Usage

Run commands from the repository root:

```bash
python main.py extract --input invoice.pdf --output ./output/
```

Extract from a scanned image:

```bash
python main.py extract --input scan.jpg --output ./output/
```

Bulk process a folder:

```bash
python main.py extract --input ./folder_of_pdfs/ --output ./output/ --bulk
```

Force key-value extraction:

```bash
python main.py extract --input receipt.png --mode keyvalue
```

Force table extraction:

```bash
python main.py extract --input report.pdf --mode table
```

Use automatic mode:

```bash
python main.py extract --input form.jpg --mode auto
```

Preview extracted tables and key-values in the terminal before saving:

```bash
python main.py extract --input invoice.pdf --output ./output/ --preview
```

Skip preprocessing for clean digital scans:

```bash
python main.py extract --input clean_scan.png --output ./output/ --no-preprocess
```

You can also run the package entry file directly:

```bash
python ocr_extractor/main.py extract --input invoice.pdf --output ./output/
```

## 📁 Project Layout

```text
ocr_extractor/
├── main.py
├── core/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── ocr_engine.py
│   ├── parser.py
│   └── exporter.py
├── utils/
│   ├── __init__.py
│   ├── pdf_handler.py
│   └── logger.py
├── samples/
├── output/
├── requirements.txt
├── README.md
└── .gitignore
```

Place sample invoices, forms, receipts, scanned tables, and PDFs in `ocr_extractor/samples/` when preparing portfolio demos. The `output/` folder is created automatically when exports are saved.

## 📤 Sample Output

Each individual input produces:

- `*_extracted_YYYYMMDD_HHMMSS.xlsx`
- `*_extracted_YYYYMMDD_HHMMSS.json`

The Excel workbook includes:

- `Summary`: file name, page count, extraction date, table count, key-value count
- `Page_1_Table_1`, `Page_2_Table_1`, etc.: one sheet per detected table
- `Key_Values`: two-column `Key | Value` layout
- `No_Tables`: created when OCR succeeds but no table structure is detected

Bulk mode additionally creates:

- `master_extraction_YYYYMMDD_HHMMSS.xlsx`

The master workbook includes a summary plus one compact sheet per processed file.

## 🛡️ Error Handling

The CLI reports clear messages for:

- Missing files or folders
- Unsupported formats
- Encrypted PDFs
- Missing Poppler installation
- Empty OCR results
- Per-file failures during bulk processing

Bulk mode continues processing remaining files even when one document fails.

## 🧪 Portfolio Demo Ideas

Add files to `ocr_extractor/samples/` such as:

- Invoice PDFs with totals, dates, and line-item tables
- Bank statement scans with transaction rows
- Form images with labels and filled values
- Receipts with noisy backgrounds
- Multi-page report PDFs with several tables

Then run:

```bash
python main.py extract --input ocr_extractor/samples/ --output ./output/ --bulk --preview
```

## Built for Fiverr Portfolio

This project is structured as a production-quality portfolio piece for OCR automation, document processing, Excel reporting, and Python CLI delivery. It demonstrates the kind of polished, client-ready workflow that can be offered as a paid Fiverr service.
