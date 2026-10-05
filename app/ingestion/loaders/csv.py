import csv
import logfire


def parse_csv(file_path: str):
    """
    Parse CSV files using the standard library csv module.
    each row is rendered as "column: value" pairs so every chunk keeps its headers for RAG.
    """

    with logfire.span("CSV Parsing", filename=file_path):
        try:
            # utf-8-sig strips the BOM that Excel adds to exported CSVs
            with open(file_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
                sample = f.read(4096)
                f.seek(0)

                # 1. Detect delimiter (comma, semicolon, tab, pipe), fall back to comma
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel

                rows = list(csv.reader(f, dialect))

            # 2. Drop fully empty rows
            rows = [row for row in rows if any(cell.strip() for cell in row)]

            if not rows:
                logfire.warning(f"No rows found in {file_path}")
                return ""

            # 3. First row is the header; fill in blank column names
            headers = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]

            # 4. Render each row as "header: value" lines, skipping empty cells
            records = []
            for row_num, row in enumerate(rows[1:], start=1):
                lines = [f"Row {row_num}"]
                for i, cell in enumerate(row):
                    value = cell.strip()
                    if not value:
                        continue
                    header = headers[i] if i < len(headers) else f"column_{i + 1}"
                    lines.append(f"{header}: {value}")
                records.append("\n".join(lines))

            full_text = "\n\n".join(records)

            if not full_text.strip():
                logfire.warning(f"CSV parser returned empty text for {file_path}")

            else:
                logfire.info(f"Successfully parsed {len(records)} rows, {len(full_text)} characters")

            return full_text

        except Exception as e:
            logfire.error(f"CSV Parse Failed: {e}")
            raise e
