import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from community import community_louvain
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots

import network

pio.templates.default = "simple_white"
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.use('Agg')


def herfindal(v):
    return np.sum(np.abs(v) ** 4)


class PlotlyDynamics:

    def __init__(self, dyn, k=None):

        self.norm_cbar = mpl.colors.Normalize(vmin=0, vmax=1)

        self.prices_label = r'$p_{i}(t)$'
        self.prods_label = r'$\gamma_{i}(t)$'
        self.stocks_label = r'$I_{ii}(t)$'

        self.budget_label = r'$\tilde{B}(t)$'
        self.cons_label = r'$C_{i}(t)$'
        self.utility_label = r'$\mathcal{U}(t)$'
        self.wage_label = r'$\frac{p_{0}(t+1)}{p_0(t)}$'

        self.profits_bar_label = r'$\overline{\mathcal{P}_i}(t)$'
        self.surplus_bar_label = r'$\overline{\mathcal{E}_i}(t)$'

        self.cmap = mpl.cm.get_cmap('jet')
        self.stocks_color = ListedColormap(sns.color_palette("PuBuGn_d", n_colors=100).as_hex())
        self.cons_color = ListedColormap(sns.color_palette("Greens_d", n_colors=100).as_hex())

        self.fig_firms_funda = None
        self.fig_house = None
        self.fig_network_raw = None
        self.fig_network_eig = None
        self.fig_firms_observ = None
        self.fig_exchanges = None

        self.rc = {"text.usetex": True,
                   "font.family": 'serif',
                   "font.size": 12,
                   "axes.grid": True,
                   "grid.alpha": 0.5,
                   "axes.xmargin": 0.025,
                   "axes.ymargin": 0.05
                   }
        self.dyn = None
        self.k = None
        self.firms = None
        if dyn:
            self.dyn = dyn
            self.k = k
            if self.k:
                self.firms = np.random.choice(self.dyn.n, self.k, replace=False) if self.k else np.arange(self.dyn.n)
            else:
                self.firms = np.arange(self.dyn.n)
            self.color_firms = np.array([self.cmap(i / self.dyn.n) for i in range(self.dyn.n)])
            self.gains, self.losses, self.supply, self.demand = self.dyn.compute_gains_losses_supplies_demand(
                self.dyn.eco,
                self.dyn.q_demand,
                self.dyn.q_exchange,
                self.dyn.prices,
                self.dyn.prods,
                self.dyn.stocks,
                self.dyn.labour)
            self.utility, self.budget = self.dyn.compute_utility_budget(self.dyn.eco,
                                                                        self.dyn.q_exchange,
                                                                        self.dyn.prices,
                                                                        self.dyn.wages,
                                                                        self.dyn.t_max,
                                                                        self.dyn.step_s,
                                                                        self.dyn.B0)
            self.diag_stocks = np.array([np.diag(stock) for stock in self.dyn.stocks])

    def update_dyn(self, dyn):
        if dyn:
            self.dyn = dyn
            self.firms = np.random.choice(self.dyn.n, self.k, replace=False) if self.k else np.arange(self.dyn.n)
            self.color_firms = np.array([self.cmap(i / self.dyn.n) for i in range(self.dyn.n)])
            self.gains, self.losses, self.supply, self.demand = self.dyn.compute_gains_losses_supplies_demand(
                self.dyn.eco,
                self.dyn.q_demand,
                self.dyn.q_exchange,
                self.dyn.prices,
                self.dyn.prods,
                self.dyn.stocks,
                self.dyn.labour)
            self.utility, self.budget = self.dyn.compute_utility_budget(self.dyn.eco,
                                                                        self.dyn.q_exchange,
                                                                        self.dyn.prices,
                                                                        self.dyn.wages,
                                                                        self.dyn.t_max,
                                                                        self.dyn.step_s,
                                                                        self.dyn.B0)
            self.diag_stocks = np.array([np.diag(stock) for stock in self.dyn.stocks])

    def run_dyn(self):
        # self.dyn.set_initial_conditions(p0, w0, g0, t1, s0, B0)
        self.dyn.discrete_dynamics()
        self.gains, self.losses, self.supply, self.demand = self.dyn.compute_gains_losses_supplies_demand(
            self.dyn.eco,
            self.dyn.q_demand,
            self.dyn.q_exchange,
            self.dyn.prices,
            self.dyn.prods,
            self.dyn.stocks,
            self.dyn.labour)
        self.utility, self.budget = self.dyn.compute_utility_budget(self.dyn.eco,
                                                                    self.dyn.q_exchange,
                                                                    self.dyn.prices,
                                                                    self.dyn.wages,
                                                                    self.dyn.t_max,
                                                                    self.dyn.step_s,
                                                                    self.dyn.B0)
        self.diag_stocks = np.array([np.diag(stock) for stock in self.dyn.stocks])

    def update_k(self, k):
        self.k = k
        self.firms = np.random.choice(self.dyn.n, self.k, replace=False) if self.k else np.arange(self.dyn.n)

    def plotHouse(self, from_eq=False):
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.02
                            )
        fig.update_xaxes(title_text=r'$t$', row=2, col=1)
        fig.update_xaxes(title_text=r'$t$', row=2, col=2)

        fig.update_yaxes(title_text=self.cons_label, row=1, col=1)
        fig.update_yaxes(title_text=self.budget_label, row=2, col=1)
        fig.update_yaxes(title_text=self.utility_label, row=1, col=2)
        fig.update_yaxes(title_text=self.wage_label, row=2, col=2)
        if from_eq:
            for firm in self.firms:
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.q_exchange[1:-1, 0, firm + 1] - self.dyn.eco.cons_eq[firm],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[firm])))
                                         ),
                              row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                     y=self.utility[1:-1] - np.dot(self.dyn.eco.house.theta,
                                                                   np.log(self.dyn.eco.cons_eq)) -
                                       self.dyn.eco.house.gamma * np.power(
                                         self.dyn.eco.labour_eq / self.dyn.eco.house.l_0, self.dyn.eco.house.phi + 1) / (
                                               self.dyn.eco.house.phi + 1),
                                     mode='lines'),
                          row=1, col=2)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                     y=self.budget[1:-1],
                                     mode='lines'),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                     y=self.dyn.wages[1:-1] - 1,
                                     mode='lines'),
                          row=2, col=2)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', exponentformat="power",
                             showexponent='last')
        else:
            for firm in self.firms:
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.q_exchange[1:-1, 0, firm + 1],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[firm])))
                                         ),
                              row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                     y=self.utility[1:-1],
                                     mode='lines'),
                          row=1, col=2)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                     y=self.budget[1:-1],
                                     mode='lines'),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                     y=self.dyn.wages[1:-1],
                                     mode='lines'),
                          row=2, col=2)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', exponentformat="power",
                             showexponent='last')

        self.fig_house = fig

    def plotNetworkEigenvalues(self):

        fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], shared_xaxes=True, shared_yaxes=True,
                            specs=[[{}, None],
                                   [{}, {}]])

        w, v = np.linalg.eig(self.dyn.eco.m_cal)
        colors = [herfindal(v[:, k]) for k in range(len(v))]
        eig_trace = go.Scattergl(x=w.real, y=w.imag, mode='markers', marker=dict(
            showscale=False,
            colorscale='Reds',
            reversescale=True,
            color=[],
            size=10,
            line_width=2))
        eig_trace.marker.color = colors
        hist_real_trace = go.Histogram(x=w.real, histnorm='probability', nbinsx=50, xaxis="x",
                                       yaxis="y3", marker_color='#c92b08')
        hist_imag_trace = go.Histogram(y=w.imag, histnorm='probability', nbinsy=50, xaxis="x2",
                                       yaxis="y", marker_color='#c92b08')

        data = [eig_trace, hist_imag_trace, hist_real_trace]
        layout = go.Layout(bargap=0,
                           bargroupgap=0,
                           xaxis=dict(title_text=r'$\Re{(\mu)}$',
                                      showgrid=True,
                                      zeroline=True,
                                      showline=True,
                                      linewidth=2,
                                      linecolor='black',
                                      mirror=True,
                                      domain=[0, 0.7]
                                      ),
                           yaxis=dict(title_text=r'$\Im{(\mu)}$',
                                      showgrid=True,
                                      zeroline=True,
                                      showline=True,
                                      linewidth=2,
                                      linecolor='black',
                                      mirror=True,
                                      domain=[0, 0.7]
                                      ),
                           xaxis2=dict(showgrid=True,
                                       showline=True,
                                       linewidth=1,
                                       linecolor='black',
                                       mirror=True,
                                       domain=[0.725, 1]
                                       ),

                           yaxis2=dict(showgrid=True,
                                       showline=True,
                                       showticklabels=False,
                                       ticks="",
                                       linewidth=1,
                                       linecolor='black',
                                       mirror=True,
                                       domain=[0, 0.7],

                                       ),
                           xaxis3=dict(showgrid=True,
                                       showline=True,
                                       showticklabels=False,
                                       ticks="",
                                       linewidth=1,
                                       linecolor='black',
                                       mirror=True,
                                       domain=[0, 0.7],

                                       ),
                           yaxis3=dict(showgrid=True,
                                       showline=True,
                                       linewidth=1,
                                       linecolor='black',
                                       mirror=True,
                                       domain=[0.725, 1]
                                       ),
                           showlegend=False
                           )

        fig = go.Figure(data=data, layout=layout)
        self.fig_network_eig = fig

    def plotNetwork(self):
        edge_x = []
        edge_y = []
        edge_width = []
        G = nx.from_numpy_matrix(self.dyn.eco.j)

        partition = community_louvain.best_partition(G)
        pos = network.community_layout(G, partition)
        # pos = nx.circular_layout(G)
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_width.append(self.dyn.eco.j[edge[0], edge[1]])

        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_z = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_z.append(self.dyn.eco.firms.z[node])
            node_text.append('Productivity factor: ' + str(self.dyn.eco.firms.z[node]))

        node_trace = go.Scattergl(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    x=1.05,
                    thickness=15,
                    title='z',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_trace.marker.color = node_z
        node_trace.text = node_text
        layout = go.Layout(showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(ticks="",
                                      showgrid=False,
                                      zeroline=False,
                                      showticklabels=False,
                                      showline=True,
                                      linewidth=2,
                                      linecolor='black',
                                      mirror=True),
                           yaxis=dict(ticks="",
                                      showgrid=False,
                                      zeroline=False,
                                      showticklabels=False,
                                      showline=True,
                                      linewidth=2,
                                      linecolor='black',
                                      mirror=True))
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

        self.fig_network_raw = fig

    def plotFirmsObserv(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02
                            )
        fig.update_xaxes(title_text=r'$t$', row=2, col=1)
        fig.update_yaxes(title_text=self.surplus_bar_label, showticksuffix='last', row=1, col=1)
        fig.update_yaxes(title_text=self.profits_bar_label, row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max - 1),
                                 y=(self.supply[1:-1, 0] - self.demand[1:-1, 0]) / (
                                         self.supply[1:-1, 0] + self.demand[1:-1, 0])
                                 ,
                                 mode='lines',
                                 line=dict(color='black', width=4, dash='dot')),
                      row=1, col=1)
        for l in self.firms:
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max - 1),
                                     y=(self.supply[1:-1, l + 1] - self.demand[1:-1, l + 1]) / (
                                             self.supply[1:-1, l + 1] + self.demand[1:-1, l + 1])
                                     ,
                                     mode='lines',
                                     marker=dict(
                                         color='rgba' + str(tuple(self.color_firms[l])))
                                     ),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max - 1),
                                     y=(self.gains[1:-1, l] - self.losses[1:-1, l]) / (
                                             self.gains[1:-1, l] + self.losses[1:-1, l]),
                                     mode='lines',
                                     marker=dict(
                                         color='rgba' + str(tuple(self.color_firms[l])))
                                     ),
                          row=2, col=1)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', exponentformat="power", showexponent='last')
        self.fig_firms_observ = fig

    def plotFirms(self, from_eq=None):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
                            )
        fig.update_xaxes(title_text=r'$t$', row=3, col=1)
        fig.update_yaxes(title_text=r'$p_{i}(t)$', row=1, col=1)
        fig.update_yaxes(title_text=r'$\gamma_{i}(t)$', row=2, col=1)
        fig.update_yaxes(title_text=r'$s_{i}(t)$', row=3, col=1)
        for l in self.firms:
            if from_eq:
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prices[1:, l] - self.dyn.eco.p_eq[l],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[l])))),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prods[1:, l] - self.dyn.eco.g_eq[l],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[l])))),
                              row=2, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.diag_stocks[1:, l],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[l])))),
                              row=3, col=1)
            else:
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prices[1:, l],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[l])))
                                         ),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prods[1:, l],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[l])))
                                         ),
                              row=2, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.diag_stocks[1:, l],
                                         mode='lines',
                                         marker=dict(
                                             color='rgba' + str(tuple(self.color_firms[l])))
                                         ),
                              row=3, col=1)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', exponentformat="power", showexponent='last')
        self.fig_firms_funda = fig

    def plotExchanges(self):
        fig_dict = {
            "data": [dict(type='heatmapgl',
                          x=np.arange(1, self.dyn.n + 1),
                          y=np.arange(1, self.dyn.n + 1),
                          z=self.dyn.Q_real[1, 1:, 1:],
                          zmin=0,
                          colorbar=dict(thickness=20, ticklen=4))],
            "layout": {'width': 700, 'height': 700},
            "frames": [dict(data=[dict(type='heatmapgl',
                                       z=self.dyn.Q_real[time, 1:, 1:],
                                       )
                                  ],
                            name=str(time),
                            )
                       for time in range(1, len(self.dyn.Q_real))]
        }
        sliders_dict = {"active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {"font": {"size": 20},
                                         "prefix": "Time:",
                                         "visible": True,
                                         "xanchor": "right"
                                         },
                        "transition": {"duration": 300,
                                       "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [{"args": [[time],
                                            {"frame": {"duration": 0,
                                                       "redraw": False},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}
                                            ],
                                   "label": str(time),
                                   "method": "animate"} for time in range(1, len(self.dyn.Q_real))]
                        }

        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig_dict["layout"]["xaxis"] = {"title": "$j$"}
        fig_dict["layout"]["yaxis"] = {"title": "$i$"}
        fig_dict["layout"]["updatemenus"] = [dict(type='buttons',
                                                  showactive=True,
                                                  y=1,
                                                  x=-0.05,
                                                  xanchor='right',
                                                  yanchor='top',
                                                  pad=dict(t=0,
                                                           r=10),
                                                  buttons=[dict(label='Play',
                                                                method='animate',
                                                                args=[None,
                                                                      dict(frame=dict(duration=500,
                                                                                      redraw=True),
                                                                           transition=dict(duration=300),
                                                                           fromcurrent=True,
                                                                           mode='immediate')]),
                                                           {
                                                               "args": [[None],
                                                                        {"frame": {"duration": 0,
                                                                                   "redraw": False},
                                                                         "mode": "immediate",
                                                                         "transition": {"duration": 0}}],
                                                               "label": "Pause",
                                                               "method": "animate"
                                                           }]
                                                  )
                                             ]

        self.fig_exchanges = go.Figure(fig_dict)



class PlotDynamics:

    def __init__(self, dyn):
        self.dyn = dyn

        self.rc = {"text.usetex": True,
                   "font.family": 'serif',
                   "font.size": 12,
                   "axes.grid": True,
                   "grid.alpha": 0.5,
                   "axes.xmargin": 0.025,
                   "axes.ymargin": 0.05,
                   }
        self.norm_cbar = mpl.colors.Normalize(vmin=0, vmax=1)

        self.prices_label = r'$\tilde{p}_{i}(t)$'
        self.prods_label = r'$\gamma_{i}(t)$'
        self.stocks_label = r'$s_{i}(t)$'

        self.budget_label = r'$\tilde{B}(t)$'
        self.cons_label = r'$C_{i}(t)$'
        self.utility_label = r'$\mathcal{U}(t)$'
        self.wage_label = r'$p_{0}(t)$'

        self.color_firms = np.array(sns.color_palette("deep", 100))
        self.stocks_color = ListedColormap(sns.color_palette("PuBuGn_d", n_colors=100).as_hex())
        self.cons_color = ListedColormap(sns.color_palette("Greens_d", n_colors=100).as_hex())

        self.fig_firms, self.axs_firms = None, None
        self.fig_house, self.axs_house = None, None

    def layout_firms(self):
        with plt.rc_context(rc=self.rc):
            self.fig_firms, self.axs_firms = plt.subplots(figsize=(8, 7),
                                                          nrows=3,
                                                          ncols=1,
                                                          tight_layout=True
                                                          )
            self.axs_firms[0].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_firms[1].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_firms[2].ticklabel_format(axis='y', scilimits=(-1, 1))

            divider0 = make_axes_locatable(self.axs_firms[0])
            cax0 = divider0.append_axes('right', size='5%', pad=0.05)
            cax0.axis('off')
            divider1 = make_axes_locatable(self.axs_firms[1])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax1.axis('off')
            divider2 = make_axes_locatable(self.axs_firms[2])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cax2.set_yticks([0, 1])
            cax2.yaxis.set_label_position('right')

            self.axs_firms[0].set_ylabel(self.prices_label)
            self.axs_firms[1].set_ylabel(self.prods_label)
            self.axs_firms[2].set_ylabel(self.stocks_label)
            self.axs_firms[-1].set_xlabel(r'$t$', rotation=0)
            cb = mpl.colorbar.ColorbarBase(cax2,
                                           cmap=self.stocks_color,
                                           norm=self.norm_cbar,
                                           ticks=[0, 1],
                                           orientation='vertical'
                                           )
            cb.set_label(r'$\sigma_{i}$', rotation=0)

    def plotFirms(self, from_eq=False, k=None):
        self.layout_firms()

        if k:
            firms = np.random.choice(self.dyn.n, k, replace=False)
            self.axs_firms[0].set_title(str(k) + r' randomly chosen firms')
        else:
            firms = np.arange(self.dyn.n)
            self.axs_firms[0].set_title(r'Every firms')

        cols = list(zip(self.stocks_color(self.dyn.eco.firms.sigma[firms]),
                        self.color_firms[firms]))
        if from_eq:
            # self.axs_firms[0].set_ylim(-2 * 10e-5, 2 * 10e-5)
            # self.axs_firms[1].set_ylim(-2 * 10e-5, 2 * 10e-5)

            p_eq = self.dyn.eco.p_eq
            g_eq = self.dyn.eco.g_eq

            for i, firm_num in enumerate(firms):
                self.axs_firms[0].plot(self.dyn.prices[1:, firm_num] - p_eq[firm_num], color=cols[i][1])
                self.axs_firms[1].plot(self.dyn.prods[1:, firm_num] - g_eq[firm_num], color=cols[i][1])
                self.axs_firms[2].plot(self.dyn.stocks[1:, firm_num], color=cols[i][0])
        else:
            for i, firm_num in enumerate(firms):
                self.axs_firms[0].plot(self.dyn.prices[1:, firm_num], color=cols[i][1])
                self.axs_firms[1].plot(self.dyn.prods[1:, firm_num], color=cols[i][1])
                self.axs_firms[2].plot(self.dyn.stocks[1:, firm_num], color=cols[i][0])

        # self.fig_firms.show()

    def layout_household(self):
        with plt.rc_context(rc=self.rc):
            self.fig_house, self.axs_house = plt.subplots(figsize=(8, 8),
                                                          nrows=2,
                                                          ncols=2,
                                                          tight_layout=True
                                                          )

            self.axs_house[0, 0].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_house[1, 0].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_house[0, 1].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_house[1, 1].ticklabel_format(axis='y', scilimits=(-1, 1))

            self.axs_house[0, 0].set_ylabel(self.cons_label)
            self.axs_house[0, 1].set_ylabel(self.utility_label)
            self.axs_house[1, 0].set_ylabel(self.budget_label)
            self.axs_house[1, 1].set_ylabel(self.wage_label)
            self.axs_house[1, 0].set_xlabel(r'$t$', rotation=0)
            self.axs_house[1, 1].set_xlabel(r'$t$', rotation=0)

            divider0 = make_axes_locatable(self.axs_house[0, 0])
            cax0 = divider0.append_axes('right', size='5%', pad=0.05)
            cax0.set_yticks([0, 1])
            cax0.yaxis.set_label_position('right')
            divider1 = make_axes_locatable(self.axs_house[0, 1])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax1.axis('off')
            divider2 = make_axes_locatable(self.axs_house[1, 0])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cax2.axis('off')
            divider3 = make_axes_locatable(self.axs_house[1, 1])
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            cax3.axis('off')

            cb = mpl.colorbar.ColorbarBase(cax0,
                                           cmap=self.cons_color,
                                           norm=self.norm_cbar,
                                           ticks=[0, 1],
                                           orientation='vertical'
                                           )
            cb.set_label(r'$\theta_{i}$', rotation=0)

    def plotHousehold(self):
        self.layout_household()

        # for i, col in enumerate(self.cons_color(self.dyn.eco.house.theta)):
        self.axs_house[0, 0].plot(self.dyn.Q_real[1:-1, 0, 1:])
        self.axs_house[0, 1].plot(self.dyn.utility[1:-1])
        self.axs_house[1, 0].plot(self.dyn.budget[1:-1])
        self.axs_house[1, 1].plot(self.dyn.wages[1:-1])

        # self.fig_firms.show()
