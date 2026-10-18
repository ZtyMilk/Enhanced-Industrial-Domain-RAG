import re
from pathlib import Path

import pymupdf

from .cleaner import is_noise_chunk


class Chunker:
    def __init__(
        self,
        input_direction: str,
        output_figure_direction: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.input_direction = input_direction
        self.output_figure_direction = output_figure_direction
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def pdf_chunker(self):
        chunks = []
        input_path = Path(self.input_direction)
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            return chunks
        print(f"{len(pdf_files)} PDFs detected")
        for pdf_file in pdf_files:
            document = pymupdf.open(str(pdf_file))
            try:
                for page_index, page in enumerate(document.pages()):
                    page_number = page_index + 1
                    raw_text = page.get_text("text")
                    captions = self._extract_figure_captions(raw_text)

                    table_chunks, table_bounding_boxes = self._extract_table(
                        page=page,
                        pdf_file=pdf_file,
                        page_number=page_number,
                    )
                    figure_chunks = self._extract_figure(
                        page=page,
                        pdf_file=pdf_file,
                        page_number=page_number,
                        captions=captions,
                    )
                    text_chunks = self._extract_text(
                        page=page,
                        pdf_file=pdf_file,
                        page_number=page_number,
                        table_bounding_boxes=table_bounding_boxes,
                    )
                    chunks.extend(table_chunks)
                    chunks.extend(figure_chunks)
                    chunks.extend(text_chunks)
            finally:
                document.close()
        return chunks

    @staticmethod
    def _extract_table(page: pymupdf.Page, pdf_file: Path, page_number: int):
        table_bounding_boxes = []
        table_chunks = []
        try:
            tables = page.find_tables()
            for table_index, table in enumerate(tables):
                markdown_table = table.to_markdown()
                if markdown_table and len(markdown_table.strip()) > 20:
                    table_bounding_boxes.append(table.bbox)
                    table_chunks.append(
                        {
                            "modality": "table",
                            "text": f"{markdown_table.strip()}",
                            "source_file": pdf_file.name,
                            "page": page_number,
                            "fileid": pdf_file.stem,
                        }
                    )
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING]One table extraction in page{page_number} failed{e}")
        return table_chunks, table_bounding_boxes

    def _extract_figure(
        self, page: pymupdf.Page, pdf_file: Path, page_number: int, captions: dict
    ):
        figure_chunks = []
        figure_path = Path(self.output_figure_direction)
        figures = page.get_images(full=True)

        for figure_index, figure_information in enumerate(figures):
            try:
                figure = page.parent.extract_image(figure_information[0])
                figure_bytes = figure["image"]
                if (
                    figure["width"] < 150
                    or figure["height"] < 150
                    or len(figure_bytes) < 5000
                ):
                    continue

                figure_save_path = (
                    figure_path
                    / f"{pdf_file.stem}_page{page_number}_fig{figure_index + 1}.{figure['ext']}"
                )
                with open(figure_save_path, "wb") as f:
                    f.write(figure_bytes)
                img_caption = captions.get(
                    f"fig_{figure_index + 1}", f"illustration on the page{page_number}"
                )
                figure_chunks.append(
                    {
                        "modality": "figure",
                        "text": f"{img_caption}",
                        "source_file": pdf_file.name,
                        "page": page_number,
                        "fileid": pdf_file.stem,
                        "figure_save_path": str(figure_save_path),
                    }
                )
            except Exception as e:  # noqa: BLE001
                print(f"[WARNING]One figure extraction in page{page_number} failed.{e}")
                continue
        return figure_chunks

    def _extract_text(
        self,
        page: pymupdf.Page,
        pdf_file: Path,
        page_number: int,
        table_bounding_boxes: list,
    ):
        text_chunks = []
        text_blocks = page.get_text_blocks()
        for text_block in text_blocks:
            bounding_box = text_block[:4]  # the form: (x0, y0, x1, y1)
            text = text_block[4].strip()
            if len(text) < 50 or re.match(r"^\d+$", text):
                continue
            if (
                re.search(r"^\s*(\d+[\.、]\s*)?根据权利要求", text)
                or len(re.findall(r"CN\s*\d+", text)) >= 2
                or "权利要求书" in text[:30]
            ) or is_noise_chunk(text=text):
                continue
            if any(
                self._is_bounding_box_overlap(bounding_box, table_bounding_box)
                for table_bounding_box in table_bounding_boxes
            ):
                continue
            cleaned_text = " ".join(text.split())
            start = 0
            text_length = len(cleaned_text)
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                final_text = cleaned_text[start:end].strip()
                if final_text:
                    text_chunks.append(
                        {
                            "modality": "text",
                            "text": final_text,
                            "source_file": pdf_file.name,
                            "page": page_number,
                            "fileid": pdf_file.stem,
                        }
                    )
                if end == text_length:
                    break
                start += max(1, self.chunk_size - self.chunk_overlap)
        return text_chunks

    @staticmethod
    def _extract_figure_captions(text: str):
        captions = {}
        pattern = r"((?:Fig(?:ure)?\.?|图)\s*\d+[\.:\s][^\n\r]+(?:\n[^\n\r]+)?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for idx, match in enumerate(matches):
            clean_cap = " ".join(match.strip().split())
            captions[f"fig_{idx + 1}"] = clean_cap
        return captions

    @staticmethod
    def _is_bounding_box_overlap(
        bounding_box_a: list[float], bounding_box_b: list[float], threshold: float = 0.5
    ):
        x0 = max(bounding_box_a[0], bounding_box_b[0])
        y0 = max(bounding_box_a[1], bounding_box_b[1])
        x1 = min(bounding_box_a[2], bounding_box_b[2])
        y1 = min(bounding_box_a[3], bounding_box_b[3])
        if x1 <= x0 or y1 <= y0:
            return False
        overlap_area = (x1 - x0) * (y1 - y0)
        area1 = (bounding_box_a[2] - bounding_box_a[0]) * (
            bounding_box_a[3] - bounding_box_a[1]
        )
        return (overlap_area / (area1 + 1e-6)) > threshold
