"""
One-off script to generate 3 sample PDFs for testing the RAG pipeline.
Run once: python data/generate_sample_pdfs.py
Each PDF has clearly distinct sections so you can verify retrieval
finds the *correct* section for a given question later.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_pdfs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
h1 = styles["Heading1"]
h2 = styles["Heading2"]
body = styles["BodyText"]


# ---------------------------------------------------------------------------
# PDF 1: Text-heavy "research paper" style doc
# ---------------------------------------------------------------------------
def generate_research_paper():
    path = os.path.join(OUTPUT_DIR, "research_paper.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = []

    story.append(Paragraph("The Impact of Remote Work on Software Team Productivity", h1))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Abstract", h2))
    story.append(Paragraph(
        "This paper examines productivity outcomes across 40 distributed software "
        "engineering teams over an 18-month period. We find that teams with structured "
        "asynchronous communication protocols showed a 23% increase in shipped features "
        "compared to teams relying primarily on synchronous meetings.", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Introduction", h2))
    story.append(Paragraph(
        "Remote work adoption accelerated dramatically starting in 2020 and has remained "
        "a dominant mode of operation for software organizations since. Prior research has "
        "focused primarily on individual productivity metrics, leaving a gap in understanding "
        "team-level dynamics. This study addresses that gap by tracking cross-functional "
        "engineering teams across three company sizes: startup (under 50 employees), "
        "mid-market (50-500 employees), and enterprise (500+ employees).", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Methodology", h2))
    story.append(Paragraph(
        "We collected data from 40 teams using a combination of git commit analysis, "
        "sprint velocity tracking, and quarterly engineer surveys. Teams were categorized "
        "by their dominant communication style: synchronous-first (daily standups, frequent "
        "video calls) versus asynchronous-first (written updates, threaded discussions, "
        "core-hours overlap of 2-3 hours only). Data was normalized for team size and "
        "seniority mix.", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Key Findings", h2))
    story.append(Paragraph(
        "Asynchronous-first teams shipped 23% more features per quarter on average, with "
        "the effect most pronounced in teams spanning more than 3 time zones. However, "
        "asynchronous-first teams also reported 15% lower scores on 'sense of team cohesion' "
        "in quarterly surveys, suggesting a tradeoff between raw output and interpersonal "
        "connection. Enterprise-size teams showed smaller productivity gains (11%) compared "
        "to startups (31%), likely due to existing process overhead in larger organizations.", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Conclusion", h2))
    story.append(Paragraph(
        "Organizations optimizing purely for throughput should consider structured "
        "asynchronous workflows, particularly for globally distributed teams. However, "
        "leaders should pair this shift with deliberate investment in team cohesion "
        "activities to offset the connection deficit observed in our survey data.", body))

    doc.build(story)
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# PDF 2: Table-heavy "quarterly report" style doc
# ---------------------------------------------------------------------------
def generate_quarterly_report():
    path = os.path.join(OUTPUT_DIR, "quarterly_report.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = []

    story.append(Paragraph("Q3 2026 Regional Sales Report", h1))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overview", h2))
    story.append(Paragraph(
        "This report summarizes sales performance across four regions for Q3 2026. "
        "Total company revenue for the quarter was $4.82 million, representing 12% "
        "growth over Q2 2026.", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Regional Sales Breakdown", h2))
    table_data = [
        ["Region", "Q3 Revenue ($)", "Units Sold", "YoY Growth"],
        ["North America", "1,920,000", "8,400", "+9%"],
        ["Europe", "1,340,000", "6,100", "+14%"],
        ["Asia Pacific", "1,120,000", "5,900", "+22%"],
        ["Latin America", "440,000", "2,200", "+6%"],
    ]
    table = Table(table_data, colWidths=[1.6 * inch, 1.5 * inch, 1.1 * inch, 1.1 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Top Products by Revenue", h2))
    product_data = [
        ["Product", "Revenue ($)", "Margin"],
        ["Product A - Pro Tier", "1,610,000", "42%"],
        ["Product B - Standard Tier", "1,280,000", "35%"],
        ["Product C - Add-on Pack", "890,000", "51%"],
    ]
    product_table = Table(product_data, colWidths=[2.2 * inch, 1.6 * inch, 1.2 * inch])
    product_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(product_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Outlook", h2))
    story.append(Paragraph(
        "Asia Pacific's 22% YoY growth makes it the fastest-growing region this quarter. "
        "The sales team recommends increasing marketing spend allocation to APAC by 15% "
        "in Q4 to capitalize on this momentum.", body))

    doc.build(story)
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# PDF 3: Headers/bullets-heavy "policy doc" style doc
# ---------------------------------------------------------------------------
def generate_policy_doc():
    path = os.path.join(OUTPUT_DIR, "employee_handbook_excerpt.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = []
    bullet_style = ParagraphStyle("Bullet", parent=body, leftIndent=18, bulletIndent=6)

    story.append(Paragraph("Employee Handbook: Remote Work Policy", h1))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Eligibility", h2))
    for line in [
        "Employees must complete a 90-day onboarding period before requesting remote status.",
        "Remote eligibility is determined by role type, not tenure alone.",
        "Engineering, Design, and Product roles are remote-eligible by default.",
        "Customer-facing Support roles require manager approval for remote status.",
    ]:
        story.append(Paragraph(f"• {line}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Core Hours", h2))
    for line in [
        "All employees must be available between 11:00 AM and 2:00 PM in their local time zone.",
        "Meetings should not be scheduled outside core hours without 48 hours notice.",
        "Teams spanning more than 4 time zones should default to asynchronous updates.",
    ]:
        story.append(Paragraph(f"• {line}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Equipment and Reimbursement", h2))
    for line in [
        "The company provides a one-time $500 home office setup stipend.",
        "Monthly internet reimbursement is capped at $50.",
        "Equipment requests above $500 require director-level approval.",
    ]:
        story.append(Paragraph(f"• {line}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Performance Expectations", h2))
    for line in [
        "Remote employees are held to the same performance standards as in-office staff.",
        "Managers must conduct monthly 1:1 check-ins with all remote reports.",
        "Quarterly OKRs apply uniformly regardless of work location.",
    ]:
        story.append(Paragraph(f"• {line}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Policy Review", h2))
    story.append(Paragraph(
        "This policy is reviewed annually by the People Operations team. Employees may "
        "submit feedback through the internal HR portal at any time.", body))

    doc.build(story)
    print(f"Created {path}")


if __name__ == "__main__":
    generate_research_paper()
    generate_quarterly_report()
    generate_policy_doc()
    print("\nAll 3 sample PDFs generated in data/sample_pdfs/")