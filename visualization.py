import numpy as np
import plotly.graph_objects as go
from gng import GrowingNeuralGas

def plot_upd(gng: GrowingNeuralGas, fig, cur_iter):
    if cur_iter % 10 != 0:
        return
    edge_x, edge_y, text = gng.get_info_for_plot()
    with fig.batch_update():
        fig.data[1].x=edge_x
        fig.data[1].y=edge_y
        
#         fig.layout['annotations'] = []100
#         for text_el in text:
#             fig.add_annotation(
#                 x=text_el[0],
#                 y=text_el[1],
#                 text=text_el[2],
#                 showarrow=False
#             )
        
        fig.data[2].x = gng.vec_weights[np.where(gng.vec_weights.mask[:, 0] == False)[0], 0]
        fig.data[2].y = gng.vec_weights[np.where(gng.vec_weights.mask[:, 0] == False)[0], 1]
        
        fig.update_layout(title='mean err - {:4.3f}, iter - {}'.format(np.mean(gng.acc_errors), cur_iter))
        
def init_fig(gng, X):
    edge_x, edge_y, text = gng.get_info_for_plot()
            
    fig = go.FigureWidget()

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                size=3
            ),
            marker_color='deepskyblue',
            name='points'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='edges'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=gng.vec_weights[np.where(gng.vec_weights.mask[:, 0] == False)[0], 0],
            y=gng.vec_weights[np.where(gng.vec_weights.mask[:, 0] == False)[0], 1],
            mode='markers',
            marker_symbol='diamond',
            marker=dict(
                size=10
            ),
            marker_color='yellowgreen',
            name='nodes'
        )
    )

    for text_el in text:
        fig.add_annotation(
            x=text_el[0],
            y=text_el[1],
            text=text_el[2],
            showarrow=False
        )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template='plotly_dark', width=1000, height=1000)
    
    return fig

def plot_err_history(gng):
    return go.Figure(
                go.Scatter(
                    x=np.arange(0, len(gng.history['err'])*gng.history_divisor, gng.history_divisor),
                    y=gng.history['err']
                ),
                layout=dict(
                    title='Err history',
                    xaxis=dict(
                        title='iteration'
                    ),
                    yaxis=dict(
                        title='err'
                    )
                )
            )