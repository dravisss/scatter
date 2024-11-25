import altair as alt
import pandas as pd
import numpy as np

# Gerando dados de exemplo
np.random.seed(42)
n_points = 200

# Criando um DataFrame com dados simulados mais complexos
data = pd.DataFrame({
    'UMAP_1': np.random.normal(0, 2, n_points),
    'UMAP_2': np.random.normal(0, 2, n_points),
    'Cluster_main': [f'Cluster {i} - Grupo Temático' for i in np.random.randint(1, 6, n_points)],
    'Size_Prob': np.random.uniform(0.1, 1.0, n_points),  # Probabilidade para tamanho
    'Temperatura': np.random.uniform(-4, 4, n_points),
    'Label': [f'Item {i}' for i in range(n_points)],
    'Description': [f'Descrição detalhada do item {i}' for i in range(n_points)],
    'Cluster_Assignments': [f'Cluster {i}, Cluster {i+1}' for i in np.random.randint(1, 4, n_points)]
})

# Extrair número do cluster para ordenação
data['Cluster_Short'] = data['Cluster_main'].apply(lambda x: x.split('-')[0].strip())

# Definindo cores para os clusters
CLUSTER_COLORS = {
    cluster: color for cluster, color in zip(
        data['Cluster_main'].unique(),
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Cores distintas
    )
}

# Calculando centroides dos clusters para labels
cluster_centroids = data.groupby('Cluster_main').agg({
    'UMAP_1': 'mean',
    'UMAP_2': 'mean'
}).reset_index()

# Criando a seleção interativa
click = alt.selection_single(
    fields=['Cluster_main'],
    on='click',
    empty='none'
)

# Configurando a opacidade baseada na seleção
opacity_condition = alt.condition(
    click,
    alt.value(1),
    alt.value(0.3)
)

# Calculando padding para centralização
x_min, x_max = data['UMAP_1'].min(), data['UMAP_1'].max()
y_min, y_max = data['UMAP_2'].min(), data['UMAP_2'].max()
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

# Criando o gráfico base
scatter_base = alt.Chart(data).mark_circle(
    stroke='lightgray',
    strokeWidth=1,
).encode(
    x=alt.X('UMAP_1:Q',
        title=None,
        scale=alt.Scale(domain=[x_min - x_padding, x_max + x_padding]),
        axis=alt.Axis(ticks=False, labels=False, grid=False)
    ),
    y=alt.Y('UMAP_2:Q',
        title=None,
        scale=alt.Scale(domain=[y_min - y_padding, y_max + y_padding]),
        axis=alt.Axis(ticks=False, labels=False, grid=False)
    ),
    size=alt.Size(
        'Size_Prob:Q',
        scale=alt.Scale(range=[50, 1000]),  # Aumentando o range para pontos maiores
        legend=None
    ),
    color=alt.Color(
        'Cluster_main:N',
        scale=alt.Scale(domain=list(CLUSTER_COLORS.keys()),
                       range=list(CLUSTER_COLORS.values())),
        legend=alt.Legend(title="Clusters", orient='right')
    ),
    opacity=opacity_condition,
    tooltip=[
        alt.Tooltip('Label:N', title='Título'),
        alt.Tooltip('Description:N', title='Descrição'),
        alt.Tooltip('Cluster_main:N', title='Cluster'),
        alt.Tooltip('Cluster_Assignments:N', title='Pertencimento'),
        alt.Tooltip('Temperatura:Q', title='Temperatura', format='.2f')
    ]
).add_selection(click)

# Criando a camada de sobreposição com opacidade reduzida
scatter_overlay = scatter_base.encode(
    opacity=alt.value(0.3)
)

# Criando a camada de labels dos clusters
text_labels = alt.Chart(cluster_centroids).mark_text(
    align='center',
    baseline='middle',
    fontSize=14,
    font='Arial',
    fontWeight='bold'
).encode(
    x='UMAP_1:Q',
    y='UMAP_2:Q',
    text='Cluster_main:N',
    opacity=alt.value(1)  # Labels sempre visíveis
)

# Combinando todas as camadas
scatter = alt.layer(
    scatter_base, 
    scatter_overlay,
    text_labels
).properties(
    width=800,  # Aumentando a largura
    height=600  # Aumentando a altura
).configure_view(
    strokeWidth=1,
    stroke='lightgray'
).configure_axis(
    grid=False
).configure_legend(
    orient='right',
    offset=10,
    titleFontSize=12,
    labelFontSize=11
).interactive()

# Para salvar como HTML:
scatter.save('scatter_plot_completo.html')
