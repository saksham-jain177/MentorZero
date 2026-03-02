import pdfplumber
import logging
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Extracts text and structured data from PDF documents.
    """
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract raw text from all pages of a PDF, with OCR fallback.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
            
        full_text = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and len(text.strip()) > 50: # Threshold for "good" text
                        full_text.append(text)
                    else:
                        # Fallback to OCR if text is sparse or missing
                        logger.info(f"Low quality text on page {page.page_number}. Attempting OCR...")
                        ocr_text = PDFParser._ocr_page(file_path, page.page_number)
                        if ocr_text:
                            full_text.append(ocr_text)
            
            return "\n\n".join(full_text)
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return ""

    @staticmethod
    def _ocr_page(file_path: str, page_number: int) -> str:
        """
        Convert a specific PDF page to an image and perform OCR.
        """
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            # Convert only the specific page to image to save memory
            images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
            if not images:
                return ""
                
            # Perform OCR
            return pytesseract.image_to_string(images[0])
        except Exception as e:
            logger.warning(f"OCR failed for page {page_number}: {e}")
            return ""

    @staticmethod
    def extract_structured_data(file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables and other structured data from PDF.
        """
        structured_data = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        structured_data.append({
                            "page": i + 1,
                            "type": "table",
                            "index": table_idx,
                            "data": table
                        })
            return structured_data
        except Exception as e:
            logger.error(f"Error extracting structured data from PDF {file_path}: {e}")
            return []

# Usage Example
if __name__ == "__main__":
    # Test with a dummy path
    parser = PDFParser()
    # text = parser.extract_text("example.pdf")
    # print(text)
