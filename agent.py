import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import io
from eda_tools import (
    get_data_description,
    generate_histogram,
    generate_scatter_plot,
    get_correlation_matrix,
    get_value_counts,
    generate_box_plot,
    perform_k_means_clustering,
    analyze_missing_data,
    get_unique_values,
    generate_value_counts_bar_chart
)

class EDAAgent:
    def __init__(self, api_key: str, df: pd.DataFrame):
        if not api_key:
            raise ValueError("API Key do Gemini não foi fornecida.")
        
        self.df = df
        genai.configure(api_key=api_key)
        
        self.tools = [
            get_data_description,
            generate_histogram,
            generate_scatter_plot,
            get_correlation_matrix,
            get_value_counts,
            generate_box_plot,
            perform_k_means_clustering,
            analyze_missing_data,
            get_unique_values,
            generate_value_counts_bar_chart
        ]
        
        column_names = ", ".join(self.df.columns)
        system_instruction = f"""
        Você é um assistente de IA especialista em Análise Exploratória de Dados.
        Sua tarefa é ajudar o usuário a entender um conjunto de dados.
        Você tem acesso a um conjunto de ferramentas para realizar análises.
        Use as ferramentas para responder às perguntas do usuário. Seja conciso e direto.
        O dataframe com o qual você está trabalhando tem as seguintes colunas: {column_names}.
        """
        
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-pro-latest',
            tools=self.tools,
            system_instruction=system_instruction
        )
        
        self.chat = self.model.start_chat()

    def interpret_plot(self, plot_figure: plt.Figure) -> str:
        """
        Converte uma figura Matplotlib em bytes e envia para o Gemini para interpretação.
        """
        buffer = io.BytesIO()
        plot_figure.savefig(buffer, format='PNG', bbox_inches='tight')
        buffer.seek(0)
        
        plot_image_part = {"mime_type": "image/png", "data": buffer.getvalue()}

        prompt = [
            "Você é um especialista em análise de dados. Sua tarefa é interpretar o gráfico a seguir. Descreva os principais insights, tendências, padrões e anomalias de forma clara e concisa em português. Use bullet points para os insights.",
            plot_image_part
        ]
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Erro ao interpretar o gráfico: {str(e)}"

    def ask(self, user_question: str):
        """
        Processa a pergunta, executa a ferramenta e, se a ferramenta gerar um gráfico,
        pede ao modelo para interpretá-lo.
        """
        try:
            response = self.chat.send_message(user_question)
            response_part = response.parts[0]
            
            if hasattr(response_part, 'function_call'):
                function_call = response_part.function_call
                tool_name = function_call.name
                tool_args = {key: value for key, value in function_call.args.items()}
                
                tool_function = next((t for t in self.tools if t.__name__ == tool_name), None)

                if tool_function:
                    tool_args['df'] = self.df
                    
                    try:
                        tool_result = tool_function(**tool_args)
                        
                        if isinstance(tool_result, plt.Figure):
                            interpretation = self.interpret_plot(tool_result)
                            return {
                                "type": "plot_with_interpretation",
                                "plot": tool_result,
                                "interpretation": interpretation
                            }
                        else:
                            return {"type": "tool_result", "data": tool_result, "tool_name": tool_name}

                    except (ValueError, TypeError) as e:
                        return {"type": "error", "data": f"Erro ao usar a ferramenta '{tool_name}': {e}"}
                else:
                    return {"type": "error", "data": f"Ferramenta '{tool_name}' não encontrada."}
            
            return {"type": "text", "data": response.text}

        except Exception as e:
            return {"type": "error", "data": f"Ocorreu um erro inesperado: {str(e)}"}
    
    def ask_conclusion(self, chat_history: list) -> str:
        """
        Usa o histórico da conversa para gerar uma conclusão sobre os dados.
        """
        if not chat_history:
            return "Não há histórico de análise para gerar uma conclusão."

        history_summary = []
        for item in chat_history:
            user_part = f"Usuário perguntou: {item['user']}"
            if item.get('type') == 'plot_with_interpretation':
                agent_part = f"Agente respondeu com um gráfico e a seguinte interpretação: {item['interpretation']}"
            else:
                agent_part = f"Agente respondeu: {item.get('agent', 'N/A')}"
            history_summary.append(f"{user_part}\n{agent_part}")
        
        conclusion_prompt = f"""
        Com base no seguinte histórico de uma sessão de análise de dados, gere uma conclusão técnica e objetiva em formato de relatório.
        Aponte os principais insights, padrões, anomalias e relações que foram descobertas.
        Seja estruturado e direto.

        Histórico da Conversa:
        {"\n---\n".join(history_summary)}

        Conclusão Final:
        """
        
        try:
            text_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = text_model.generate_content(conclusion_prompt)
            return response.text
        except Exception as e:
            return f"Erro ao gerar conclusão: {str(e)}"
