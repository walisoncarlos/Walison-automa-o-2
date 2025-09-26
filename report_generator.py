from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime
import io

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Relatório de Análise de Dados', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        # O 'latin-1' replace é para evitar erros com caracteres especiais
        self.multi_cell(0, 5, body.encode('latin-1', 'replace').decode('latin-1'))
        self.ln()
        
    def add_plot(self, plot_fig: plt.Figure):
        buffer = io.BytesIO()
        plot_fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        page_width = self.w - 2 * self.l_margin
        self.image(buffer, x=self.l_margin, w=page_width)
        self.ln(5)
        buffer.close()


def create_pdf_report(
    chat_history: list, 
    conclusion: str, 
    file_name: str, 
    initial_report_data: dict
) -> bytes:
    """
    Gera um relatório PDF completo.
    """
    pdf = PDFReport()
    pdf.add_page()

    pdf.chapter_title('1. Resumo da Análise')
    pdf.chapter_body(f"Arquivo Analisado: {file_name}")
    analysis_date = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
    pdf.chapter_body(f"Data da Análise: {analysis_date}")

    pdf.chapter_title('2. Conclusão Principal Gerada pela IA')
    pdf.chapter_body(conclusion)

    if initial_report_data:
        pdf.add_page()
        pdf.chapter_title('3. Análise Automática Inicial')
        for section, results in initial_report_data.items():
            pdf.set_font('Arial', 'B', 11)
            pdf.multi_cell(0, 5, section)
            pdf.ln(2)
            for result in results:
                if result['type'] == 'text':
                    pdf.set_font('Arial', '', 11)
                    clean_text = result['content'].replace('####', '').replace('```', '')
                    pdf.chapter_body(clean_text)
                elif result['type'] == 'plot':
                    pdf.add_plot(result['content'])
                pdf.ln(5)

    if chat_history:
        pdf.add_page()
        pdf.chapter_title('4. Análise Detalhada (Histórico do Chat)')
        for message in chat_history:
            pdf.set_font('Arial', 'B', 11)
            pdf.chapter_body(f"Usuário: {message['user']}")
            
            pdf.set_font('Arial', '', 11)
            response_type = message.get("type")
            
            if response_type == "plot_with_interpretation":
                pdf.chapter_body("Agente: (Gráfico gerado abaixo com interpretação)")
                pdf.add_plot(message['plot'])
                pdf.ln(2)
                pdf.set_font('Arial', 'I', 10)
                pdf.chapter_body(f"Interpretação do Agente:\n{message['interpretation']}")
            else:
                 pdf.chapter_body(f"Agente: {message.get('agent', 'N/A')}")
            pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')
