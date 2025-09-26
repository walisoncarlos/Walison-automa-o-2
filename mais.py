import streamlit as st
import pandas as pd
from agent import EDAAgent
import matplotlib.pyplot as plt
from report_generator import create_pdf_report
from eda_tools import (
    get_data_description,
    analyze_missing_data,
    generate_histogram,
    generate_box_plot,
    generate_value_counts_bar_chart,
    get_correlation_matrix
)

# --- Configuração da Página ---
st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="wide"
)

# --- Funções Auxiliares ---
def initialize_session_state():
    """Inicializa o estado da sessão."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'conclusion' not in st.session_state:
        st.session_state.conclusion = None
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    if 'initial_report_data' not in st.session_state:
        st.session_state.initial_report_data = None

def display_chat_history():
    """Exibe o histórico do chat na interface."""
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            response_type = message.get("type")
            
            if response_type == "plot_with_interpretation":
                st.pyplot(message["plot"])
                plt.close(message["plot"])
                st.markdown(message["interpretation"])
            elif response_type in ["text", "error"]:
                st.markdown(message.get("agent", ""))

@st.cache_data
def run_automatic_analysis(_df):
    """Executa um conjunto de análises básicas e retorna os resultados."""
    if _df is None:
        return None
        
    report = {}
    
    report['Visão Geral e Dados Faltantes'] = [
        {'type': 'text', 'content': get_data_description(_df)},
        {'type': 'text', 'content': analyze_missing_data(_df)}
    ]

    numeric_cols = _df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        report['Análise de Colunas Numéricas'] = []
        for col in numeric_cols:
            report['Análise de Colunas Numéricas'].append({'type': 'plot', 'content': generate_histogram(_df, col)})
            report['Análise de Colunas Numéricas'].append({'type': 'plot', 'content': generate_box_plot(_df, col)})
        
        if len(numeric_cols) > 1:
            report['Análise de Correlação'] = [{'type': 'plot', 'content': get_correlation_matrix(_df)}]

    categorical_cols = _df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        report['Análise de Colunas Categóricas'] = []
        for col in categorical_cols:
             if _df[col].nunique() <= 20:
                report['Análise de Colunas Categóricas'].append({'type': 'plot', 'content': generate_value_counts_bar_chart(_df, col)})

    return report

# --- Interface Principal ---
st.title("🤖 Agente Proativo de Análise de Dados (E.D.A.)")
initialize_session_state()

# --- Barra Lateral ---
with st.sidebar:
    st.header("⚙️ Configuração")
    api_key = st.text_input("Chave da API do Gemini", type="password", help="Obtenha sua chave no Google AI Studio.")
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=["csv"])

    if uploaded_file and not st.session_state.df:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            if api_key:
                st.session_state.agent = EDAAgent(api_key=api_key, df=df)
            else:
                st.warning("Por favor, insira sua chave da API do Gemini para ativar o agente.")
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            st.session_state.df = None

    if st.session_state.agent:
        st.header("🎯 Ações")
        if st.button("Gerar Conclusão Final"):
            if st.session_state.chat_history:
                with st.spinner("Gerando conclusão..."):
                    conclusion_text = st.session_state.agent.ask_conclusion(st.session_state.chat_history)
                    st.session_state.conclusion = conclusion_text
                    st.success("Conclusão gerada! Você já pode baixar o relatório.")
            else:
                st.warning("Faça algumas perguntas no chat antes de gerar a conclusão.")
        
        st.download_button(
            label="📄 Baixar Relatório Completo em PDF",
            data=create_pdf_report(
                st.session_state.chat_history, 
                st.session_state.conclusion,
                st.session_state.file_name,
                st.session_state.initial_report_data
            ) if st.session_state.conclusion else b"",
            file_name=f"relatorio_analise_{st.session_state.file_name}.pdf" if st.session_state.file_name else "relatorio.pdf",
            mime="application/pdf",
            disabled=not st.session_state.conclusion,
            help="Clique em 'Gerar Conclusão Final' primeiro para habilitar o download."
        )

# --- Área de Conteúdo Principal ---
if st.session_state.agent:
    if st.session_state.initial_report_data is None:
        with st.spinner("O agente está realizando a análise automática inicial..."):
            st.session_state.initial_report_data = run_automatic_analysis(st.session_state.df)

    if st.session_state.initial_report_data:
        st.header("🔍 Painel de Análise Automática Inicial")
        for section, results in st.session_state.initial_report_data.items():
            with st.expander(f"**{section}**", expanded=(section == 'Visão Geral e Dados Faltantes')):
                for result in results:
                    if result['type'] == 'text':
                        st.markdown(result['content'])
                    elif result['type'] == 'plot':
                        st.pyplot(result['content'])
                        plt.close(result['content'])
    
    st.header("💬 Converse com o Agente")
    display_chat_history()
    user_prompt = st.chat_input("Faça perguntas específicas ou peça análises mais profundas...")
    if user_prompt:
        st.session_state.chat_history.append({"user": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando e interpretando..."):
                response = st.session_state.agent.ask(user_prompt)
                
                if response["type"] == "plot_with_interpretation":
                    st.session_state.chat_history[-1].update({
                        "type": "plot_with_interpretation",
                        "plot": response["plot"],
                        "interpretation": response["interpretation"]
                    })
                else:
                    st.session_state.chat_history[-1].update({
                        "agent": response.get("data", ""), 
                        "type": response.get("type", "error")
                    })
            st.rerun()
else:
    st.info("👋 Para começar, insira sua chave da API e carregue um arquivo CSV na barra lateral.")
