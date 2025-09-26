import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ferramenta para obter uma visão geral do dataframe
def get_data_description(df: pd.DataFrame) -> str:
    """
    Retorna uma descrição geral dos dados, incluindo informações sobre as colunas,
    tipos de dados e estatísticas descritivas.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    numeric_df = df.select_dtypes(include=['number'])
    desc = numeric_df.describe().to_string() if not numeric_df.empty else "Nenhuma coluna numérica para descrever."
    
    return f"#### Informações Gerais das Colunas:\n```\n{info_str}\n```\n\n#### Estatísticas Descritivas (Numéricas):\n```\n{desc}\n```"

# Ferramenta para gerar um histograma de uma coluna
def generate_histogram(df: pd.DataFrame, column_name: str) -> plt.Figure:
    """
    Gera e retorna um histograma para uma coluna numérica específica.
    """
    if column_name not in df.columns:
        raise ValueError(f"Coluna '{column_name}' não encontrada no DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"A coluna '{column_name}' não é numérica e não pode ser usada para um histograma.")
    
    fig, ax = plt.subplots()
    sns.histplot(df[column_name].dropna(), kde=True, ax=ax)
    ax.set_title(f'Distribuição de {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frequência')
    return fig

# Ferramenta para gerar um gráfico de dispersão (scatter plot)
def generate_scatter_plot(df: pd.DataFrame, column_x: str, column_y: str) -> plt.Figure:
    """
    Gera e retorna um gráfico de dispersão para visualizar a relação entre duas colunas numéricas.
    """
    for col in [column_x, column_y]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"A coluna '{col}' não é numérica. Gráficos de dispersão requerem dados numéricos.")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=column_x, y=column_y, ax=ax)
    ax.set_title(f'Relação entre {column_x} e {column_y}')
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)
    return fig

# Ferramenta para obter a matriz de correlação
def get_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Calcula e retorna um heatmap da matriz de correlação para as colunas numéricas.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        raise ValueError("A matriz de correlação requer pelo menos duas colunas numéricas.")
        
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Matriz de Correlação')
    return fig

# Ferramenta para contar valores de uma coluna (texto)
def get_value_counts(df: pd.DataFrame, column_name: str) -> str:
    """
    Retorna a contagem de valores únicos para uma coluna específica em formato de texto.
    """
    if column_name not in df.columns:
        raise ValueError(f"Coluna '{column_name}' não encontrada no DataFrame.")
        
    counts = df[column_name].value_counts().to_string()
    return f"#### Contagem de valores para a coluna '{column_name}':\n```\n{counts}\n```"

# Ferramenta para detecção de outliers
def generate_box_plot(df: pd.DataFrame, column_name: str) -> plt.Figure:
    """
    Gera um box plot para uma coluna numérica, que é excelente para visualizar outliers.
    """
    if column_name not in df.columns:
        raise ValueError(f"Coluna '{column_name}' não encontrada.")
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"A coluna '{column_name}' não é numérica e não pode ser usada para um box plot.")
        
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column_name], ax=ax)
    ax.set_title(f'Box Plot de {column_name} (Análise de Outliers)')
    return fig

# Ferramenta para análise de clusters
def perform_k_means_clustering(df: pd.DataFrame, column_x: str, column_y: str, n_clusters: int = 3) -> plt.Figure:
    """
    Realiza clusterização K-Means com duas colunas numéricas e exibe o resultado.
    """
    for col in [column_x, column_y]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"A coluna '{col}' não é numérica e não pode ser usada para clusterização.")

    data_to_cluster = df[[column_x, column_y]].dropna()
    if len(data_to_cluster) < n_clusters:
        raise ValueError("Não há dados suficientes para formar os clusters solicitados.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    data_to_cluster['cluster'] = clusters
    fig, ax = plt.subplots()
    sns.scatterplot(data=data_to_cluster, x=column_x, y=column_y, hue='cluster', palette='viridis', ax=ax)
    ax.set_title(f'Clusterização K-Means ({n_clusters} clusters)')
    return fig

# Ferramenta para analisar dados faltantes (missing values)
def analyze_missing_data(df: pd.DataFrame) -> str:
    """
    Verifica e retorna a contagem de dados faltantes (nulos) para cada coluna.
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if missing_data.empty:
        return "Ótima notícia! Não foram encontrados dados faltantes em nenhuma coluna."
    
    report = "#### Análise de Dados Faltantes:\n"
    report += "A contagem de valores nulos por coluna é a seguinte:\n```\n"
    report += missing_data.to_string()
    report += "\n```"
    return report

# Ferramenta para obter valores únicos
def get_unique_values(df: pd.DataFrame, column_name: str) -> str:
    """
    Retorna os valores únicos e a contagem de valores únicos de uma coluna específica.
    """
    if column_name not in df.columns:
        raise ValueError(f"Coluna '{column_name}' não encontrada no DataFrame.")
    
    unique_values = df[column_name].unique()
    num_unique_values = df[column_name].nunique()
    
    report = f"#### Análise de Valores Únicos para a Coluna '{column_name}':\n"
    report += f"- **Número de valores únicos:** {num_unique_values}\n"
    
    if num_unique_values <= 20:
        report += f"- **Valores:** `{[str(val) for val in unique_values]}`"
    else:
        report += f"- **Amostra de valores:** `{[str(val) for val in unique_values[:5]]}` (e mais {num_unique_values-5})"
    return report

# Ferramenta para gerar gráfico de barras de contagem de valores
def generate_value_counts_bar_chart(df: pd.DataFrame, column_name: str) -> plt.Figure:
    """
    Gera um gráfico de barras com a contagem de cada valor em uma coluna categórica.
    """
    if column_name not in df.columns:
        raise ValueError(f"Coluna '{column_name}' não encontrada.")
    if pd.api.types.is_numeric_dtype(df[column_name]):
        if df[column_name].nunique() > 20:
             raise TypeError(f"A coluna '{column_name}' é numérica com muitos valores. Use um histograma.")
    
    counts = df[column_name].value_counts().nlargest(15)
    
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f'Contagem de Valores para {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Contagem')
    plt.xticks(rotation=45, ha='right')
    return fig
