from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(
    total,
    anomalies,
    critical,
    filename="report.pdf"
):

    doc = SimpleDocTemplate(
        filename
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AI Powered Production Anomaly Detector",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            f"Total Records: {total}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Anomalies: {anomalies}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Critical: {critical}",
            styles["BodyText"]
        )
    )

    doc.build(content)

    return filename