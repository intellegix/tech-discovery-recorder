"""
Premium PDF generation service for Tech Discovery Recorder reports.
Creates professional-quality PDF reports with enhanced structured analysis.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from ..models.schemas import StructuredNotes
from ..utils.result import Result, Ok, Err, ProcessingError
from ..config.settings import settings

logger = logging.getLogger(__name__)


class PremiumPDFGenerator:
    """Premium PDF report generator for Tech Discovery analysis."""

    def __init__(self):
        """Initialize PDF generator with professional styling."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text to prevent garbled characters in PDF generation.

        Args:
            text: Input text that may contain problematic characters

        Returns:
            Cleaned text safe for PDF generation
        """
        if not text:
            return ""

        # Convert to string if not already
        text = str(text)

        # Replace problematic Unicode characters that cause garbled text
        replacements = {
            # Smart quotes and dashes
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '--', '…': '...',

            # Bullet points and special symbols
            '•': '* ', '◦': '- ', '▪': '- ', '▫': '- ', '‣': '> ',

            # Mathematical symbols
            '×': 'x', '÷': '/', '±': '+/-',

            # Other problematic characters
            '®': '(R)', '©': '(C)', '™': '(TM)',

            # Remove zero-width characters and other control chars
            '\u200b': '', '\u200c': '', '\u200d': '', '\ufeff': '',
        }

        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove any remaining non-printable characters except basic whitespace
        import re
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _setup_custom_styles(self):
        """Setup custom professional styles for the PDF."""
        # Main title style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f2937'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#374151'),
            fontName='Helvetica-Bold'
        ))

        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#4b5563'),
            fontName='Helvetica-Bold'
        ))

        # Executive summary style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=15,
            textColor=colors.HexColor('#1f2937'),
            alignment=TA_JUSTIFY,
            firstLineIndent=0.25*inch,
            leftIndent=0.25*inch,
            rightIndent=0.25*inch,
            borderWidth=1,
            borderColor=colors.HexColor('#d1d5db'),
            borderPadding=10
        ))

        # Action item style
        self.styles.add(ParagraphStyle(
            name='ActionItem',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=8,
            spaceAfter=4,
            leftIndent=0.25*inch,
            bulletIndent=0.1*inch
        ))

        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER
        ))

    def generate_premium_report(
        self,
        transcript: str,
        structured_notes: StructuredNotes,
        metadata: Dict[str, Any],
        output_path: str,
        recording_name: str = "Discovery Session"
    ) -> Result[str, ProcessingError]:
        """
        Generate a premium PDF report from structured analysis.

        Args:
            transcript: Full audio transcript
            structured_notes: Enhanced structured analysis
            metadata: Processing metadata
            output_path: Path for output PDF file
            recording_name: Name of the recording session

        Returns:
            Result with PDF file path or error
        """
        try:
            logger.info(f"Generating premium PDF report: {output_path}")

            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )

            # Build document content
            story = []

            # Add header and title
            self._add_header(story, recording_name, metadata)

            # Add executive summary
            self._add_executive_summary(story, structured_notes)

            # Add main sections
            self._add_action_items(story, structured_notes)
            self._add_decisions(story, structured_notes)
            self._add_architecture_details(story, structured_notes)
            self._add_technical_specifications(story, structured_notes)
            self._add_risks_opportunities(story, structured_notes)
            self._add_follow_ups(story, structured_notes)

            # Add full transcript
            self._add_transcript(story, transcript)

            # Add footer
            self._add_footer(story, metadata)

            # Build PDF
            doc.build(story)

            logger.info(f"Premium PDF report generated successfully: {output_path}")
            return Ok(output_path)

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return Err(ProcessingError(f"PDF generation failed: {str(e)}"))

    def _add_header(self, story: List, recording_name: str, metadata: Dict[str, Any]):
        """Add professional header to the report."""
        # Company/Product header
        header_table = Table([
            ["Tech Discovery Recorder", "Professional Intelligence Report"],
            ["by Austin Kidwell | Intellegix", ""]
        ], colWidths=[3*inch, 3*inch])

        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 12),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 10),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#6b7280')),
            ('FONTNAME', (0, 1), (0, 1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (0, 1), 9),
            ('TEXTCOLOR', (0, 1), (0, 1), colors.HexColor('#6b7280')),
        ]))

        story.append(header_table)
        story.append(Spacer(1, 0.3*inch))

        # Main title
        story.append(Paragraph(recording_name, self.styles['MainTitle']))

        # Date and metadata
        generation_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated on {generation_date}", self.styles['Metadata']))

        # Processing summary
        if metadata:
            duration = metadata.get('audio_duration_seconds', 0)
            if duration:
                duration_str = f"{duration/60:.1f} minutes" if duration > 60 else f"{duration:.0f} seconds"
                story.append(Paragraph(f"Audio Duration: {duration_str}", self.styles['Metadata']))

        story.append(Spacer(1, 0.4*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 0.3*inch))

    def _add_executive_summary(self, story: List, structured_notes: StructuredNotes):
        """Add executive summary section."""
        if hasattr(structured_notes, 'executive_summary') and structured_notes.executive_summary:
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            story.append(Paragraph(structured_notes.executive_summary, self.styles['ExecutiveSummary']))
            story.append(Spacer(1, 0.2*inch))

    def _add_action_items(self, story: List, structured_notes: StructuredNotes):
        """Add action items section with FIXED text formatting to prevent garbled text."""
        if hasattr(structured_notes, 'action_items') and structured_notes.action_items:
            story.append(Paragraph("Action Items", self.styles['SectionHeader']))

            # Use simple list format instead of complex table to avoid garbled text
            for i, item in enumerate(structured_notes.action_items[:8], 1):  # Limit to 8 items
                if isinstance(item, dict):
                    priority = self._sanitize_text(str(item.get('priority', 'Medium')))
                    task = self._sanitize_text(str(item.get('task', item.get('description', str(item)))))
                    owner = self._sanitize_text(str(item.get('owner', 'TBD')))

                    # Limit task length to prevent overflow
                    if len(task) > 120:
                        task = task[:117] + "..."

                    item_text = f"<b>{i}. [{priority}] {task}</b>"
                    if owner and owner != 'TBD':
                        item_text += f"<br/><i>Owner: {owner}</i>"

                else:
                    clean_item = self._sanitize_text(str(item))
                    if len(clean_item) > 120:
                        clean_item = clean_item[:117] + "..."
                    item_text = f"<b>{i}. {clean_item}</b>"

                story.append(Paragraph(item_text, self.styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

            story.append(Spacer(1, 0.3*inch))

    def _add_decisions(self, story: List, structured_notes: StructuredNotes):
        """Add decisions section with fixed text handling."""
        if hasattr(structured_notes, 'decisions') and structured_notes.decisions:
            story.append(Paragraph("Key Decisions", self.styles['SectionHeader']))

            for i, decision in enumerate(structured_notes.decisions[:5], 1):
                if isinstance(decision, dict):
                    decision_text = self._sanitize_text(str(decision.get('decision', str(decision))))
                    rationale = self._sanitize_text(str(decision.get('rationale', '')))
                    impact = self._sanitize_text(str(decision.get('impact', '')))

                    # Limit text length to prevent overflow
                    if len(decision_text) > 150:
                        decision_text = decision_text[:147] + "..."
                    if len(rationale) > 200:
                        rationale = rationale[:197] + "..."
                    if len(impact) > 200:
                        impact = impact[:197] + "..."

                    story.append(Paragraph(f"<b>{i}. {decision_text}</b>", self.styles['Normal']))
                    if rationale:
                        story.append(Paragraph(f"<i>Rationale:</i> {rationale}", self.styles['ActionItem']))
                    if impact:
                        story.append(Paragraph(f"<i>Impact:</i> {impact}", self.styles['ActionItem']))
                else:
                    clean_decision = self._sanitize_text(str(decision))
                    if len(clean_decision) > 150:
                        clean_decision = clean_decision[:147] + "..."
                    story.append(Paragraph(f"<b>{i}. {clean_decision}</b>", self.styles['Normal']))

                story.append(Spacer(1, 0.1*inch))

            story.append(Spacer(1, 0.2*inch))

    def _add_architecture_details(self, story: List, structured_notes: StructuredNotes):
        """Add architecture details section."""
        if hasattr(structured_notes, 'architecture_details') and structured_notes.architecture_details:
            story.append(Paragraph("Architecture Details", self.styles['SectionHeader']))

            for i, detail in enumerate(structured_notes.architecture_details[:5], 1):
                if isinstance(detail, dict):
                    component = detail.get('component', 'Component')
                    description = detail.get('description', str(detail))
                    trade_offs = detail.get('trade_offs', '')

                    story.append(Paragraph(f"<b>{i}. {component}</b>", self.styles['SubsectionHeader']))
                    story.append(Paragraph(description, self.styles['Normal']))
                    if trade_offs:
                        story.append(Paragraph(f"<i>Trade-offs:</i> {trade_offs}", self.styles['ActionItem']))
                else:
                    story.append(Paragraph(f"<b>{i}. {detail}</b>", self.styles['Normal']))

                story.append(Spacer(1, 0.1*inch))

            story.append(Spacer(1, 0.2*inch))

    def _add_technical_specifications(self, story: List, structured_notes: StructuredNotes):
        """Add technical specifications section."""
        if hasattr(structured_notes, 'technical_specs') and structured_notes.technical_specs:
            story.append(Paragraph("Technical Specifications", self.styles['SectionHeader']))

            for i, spec in enumerate(structured_notes.technical_specs[:5], 1):
                if isinstance(spec, dict):
                    requirement = spec.get('requirement', spec.get('spec', str(spec)))
                    category = spec.get('category', '')
                    acceptance = spec.get('acceptance_criteria', '')

                    story.append(Paragraph(f"<b>{i}. {requirement}</b>", self.styles['Normal']))
                    if category:
                        story.append(Paragraph(f"<i>Category:</i> {category}", self.styles['ActionItem']))
                    if acceptance:
                        story.append(Paragraph(f"<i>Acceptance:</i> {acceptance}", self.styles['ActionItem']))
                else:
                    story.append(Paragraph(f"<b>{i}. {spec}</b>", self.styles['Normal']))

                story.append(Spacer(1, 0.1*inch))

            story.append(Spacer(1, 0.2*inch))

    def _add_risks_opportunities(self, story: List, structured_notes: StructuredNotes):
        """Add risks and opportunities section."""
        if hasattr(structured_notes, 'risks_concerns') and structured_notes.risks_concerns:
            story.append(Paragraph("Risks & Opportunities", self.styles['SectionHeader']))

            for i, item in enumerate(structured_notes.risks_concerns[:5], 1):
                if isinstance(item, dict):
                    risk = item.get('risk', item.get('opportunity', str(item)))
                    mitigation = item.get('mitigation', item.get('value', ''))
                    probability = item.get('probability', '')

                    story.append(Paragraph(f"<b>{i}. {risk}</b>", self.styles['Normal']))
                    if probability:
                        story.append(Paragraph(f"<i>Probability:</i> {probability}", self.styles['ActionItem']))
                    if mitigation:
                        story.append(Paragraph(f"<i>Strategy:</i> {mitigation}", self.styles['ActionItem']))
                else:
                    story.append(Paragraph(f"<b>{i}. {item}</b>", self.styles['Normal']))

                story.append(Spacer(1, 0.1*inch))

            story.append(Spacer(1, 0.2*inch))

    def _add_follow_ups(self, story: List, structured_notes: StructuredNotes):
        """Add follow-ups section."""
        if hasattr(structured_notes, 'follow_ups') and structured_notes.follow_ups:
            story.append(Paragraph("Follow-up Actions", self.styles['SectionHeader']))

            for i, followup in enumerate(structured_notes.follow_ups[:10], 1):
                if isinstance(followup, dict):
                    item = followup.get('item', str(followup))
                    urgency = followup.get('urgency', '')
                    effort = followup.get('estimated_effort', '')

                    story.append(Paragraph(f"{i}. {item}", self.styles['ActionItem']))
                    if urgency or effort:
                        details = f"Urgency: {urgency}" if urgency else ""
                        if effort:
                            details += f" | Effort: {effort}" if details else f"Effort: {effort}"
                        story.append(Paragraph(f"<i>{details}</i>", self.styles['ActionItem']))
                else:
                    story.append(Paragraph(f"{i}. {followup}", self.styles['ActionItem']))

            story.append(Spacer(1, 0.2*inch))

    def _add_transcript(self, story: List, transcript: str):
        """Add full transcript section with fixed text handling."""
        if transcript:
            story.append(PageBreak())
            story.append(Paragraph("Complete Transcript", self.styles['SectionHeader']))

            # Clean and sanitize the transcript
            clean_transcript = self._sanitize_text(transcript)

            # Format transcript with proper paragraphs
            transcript_paragraphs = clean_transcript.split('\n\n') if '\n\n' in clean_transcript else [clean_transcript]

            for para in transcript_paragraphs:  # Show complete transcript
                if para.strip():
                    # Break long paragraphs into smaller chunks at sentence boundaries
                    clean_para = para.strip()

                    # If paragraph is very long, break it into chunks
                    if len(clean_para) > 2000:
                        sentences = clean_para.split('. ')
                        current_chunk = ""

                        for sentence in sentences:
                            if len(current_chunk + sentence) > 1800:
                                if current_chunk:
                                    story.append(Paragraph(current_chunk + '.', self.styles['Normal']))
                                    story.append(Spacer(1, 0.1*inch))
                                current_chunk = sentence
                            else:
                                current_chunk += ('. ' if current_chunk else '') + sentence

                        # Add the final chunk
                        if current_chunk:
                            story.append(Paragraph(current_chunk + ('.' if not current_chunk.endswith('.') else ''), self.styles['Normal']))
                            story.append(Spacer(1, 0.1*inch))
                    else:
                        story.append(Paragraph(clean_para, self.styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))

    def _add_footer(self, story: List, metadata: Dict[str, Any]):
        """Add footer with metadata."""
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e5e7eb')))

        # Processing info
        footer_text = "Generated by Tech Discovery Recorder | Enhanced with Claude AI Analysis"
        story.append(Paragraph(footer_text, self.styles['Metadata']))

        if metadata:
            model_used = metadata.get('claude_model_used', 'claude-3-haiku-20240307')
            story.append(Paragraph(f"AI Model: {model_used} | Whisper Transcription", self.styles['Metadata']))


# Global instance
pdf_generator = PremiumPDFGenerator()