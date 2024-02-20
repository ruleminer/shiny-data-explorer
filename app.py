from pathlib import Path
from shiny import App, render, ui, Inputs, Outputs, Session, reactive, req
from shiny.types import FileInfo
import shinyswatch
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#import pyodide_js                   #package for ShinyLive
#pyodide_js.loadPackage("scipy")

file_path = Path("C:/Users/kamil/OneDrive - Politechnika Śląska/Lukasiewicz/staz/staz/bezrobocie_data.csv") #path to example data in local file
#file_path = Path(__file__).parent / "bezrobocie_data.csv" #path to example data in ShinyLive

app_ui = ui.page_navbar(
    shinyswatch.theme.united(),
    
    ui.nav("Zbiór danych",
        ui.row(
            ui.column(6,
                      ui.input_file("file1", "Wybierz plik:", accept=[".csv"], multiple=False, 
                                    button_label = "Szukaj...",
                                    placeholder = "Nie wybrano pliku")),
            ui.column(6,
                      ui.tags.p("Wykorzystaj przykładowe dane"),
                      ui.input_switch("ready_data", "Przykładowe dane")),
        ),
    
        ui.input_select(
            "number_of_rows",
            "Ilość wyświetlanych wierszy",
            {
                "10": "10",
                "20": "20",
                "50": "50",
                "100": "100",
            },
        ),
        ui.output_data_frame("summary_data"),

    ),
    ui.nav("Opis zbioru",
        ui.navset_tab(
            ui.nav(
            "Charakterystyka zbioru",
                ui.row(
                    ui.column(6,
                        "Charakterystyka zbioru",
                        ui.output_data_frame("charakterystyka_zbioru")
                    ),
                    ui.column(6,
                        "Procent brakujących wartości",
                        ui.output_data_frame("wartosci_nan_ch"),
                    ),
                ),
            ),
            ui.nav(
                "Podstawowe statystyki",
                ui.navset_pill(
                    ui.nav("Statystyki atrybutów numerycznych",
                        ui.input_selectize("numeric_stat", "Wybierz zmienną numeryczną", [], multiple=True),
                        ui.output_data_frame("num_stats"),
                    ),
                    ui.nav("Statystyki atrybutów nominalnych",
                              ui.input_selectize("category_stat", "Wybierz kategorię", [], multiple=True),
                              ui.output_data_frame("atr_stats")
                    ),
                ),
            ),
        ),
    ),
    ui.nav("Wizualizacje",
           ui.navset_tab(
                ui.nav("Atrybuty numeryczne",
                       ui.row(
                            ui.tags.h1("Histogram"),
                            ui.input_select("numeric_hist", "Wybierz zmienną numeryczną", []),
                            ui.input_slider("bins", "Ilość binów", 2, 30, 15),
                            output_widget("histogram"),
                        ),
                       ui.row(
                           
                            ui.tags.h1("Wykres pudełkowy"),
                            ui.input_selectize("numeric_box", "Wybierz zmienną", [], multiple=True),
                            output_widget("boxplot"),
                        ),
                ),
                ui.nav("Atrybuty nominalne", 
                    ui.input_select("category_bar", "Wybierz kategorię", []),
                    output_widget("barplot_atr")
                    ),
            ),
    ),
    ui.nav("Korelacje",
              ui.row(
                ui.column(4,
                             ui.input_select("numeric_corr", "Wybierz pierwszą zmienną", []),
                             ),
                ui.column(4,
                             ui.input_select("numeric_corr2", "Wybierz drugą zmienną", []),
                             ),
                ui.column(4,
                            ui.input_select("method_corr", "Wybierz metodę", {"pearson":"Pearson",
                                                                            "spearman":"Spearman",
                                                                            "kendall":"Kenall"}),
                        ),
                output_widget("correlation"),
                ui.output_data_frame("corr_df"),
                ),
    ),
    ui.nav("Szeregi czasowe",
           ui.row(
               ui.column(4,
                    ui.input_select("time", "Wybierz szereg czasowy", []),
               ),
               ui.column(4,
                    ui.input_select("variable", "Wybierz zmienną", []),
               ),
               ui.column(4,
                    ui.input_select("category_time", "Wybierz kategorie", []),
               ),
           ),
            output_widget("time_city"),
    ),
    
    ui.nav("Agregaty", 
        ui.row(  
            ui.column(6,
                ui.input_select("category_agg", "Wybierz kategorię", []),
                ui.input_select("numeric_agg", "Wybierz zmienną numeryczną", []),
            ),
            ui.column(6, 
                ui.input_checkbox_group("statistic", "Wybierz metryki",            
                    {
                        "min": "Min",
                        "max": "Max",
                        "mean": "Średnia",
                        "median": "Mediana",
                        "q1": "Górny kwartyl",
                        "q3": "Dolny kwartyl",
                    },
                ),
            ),
        ),
        ui.output_data_frame("grouped_data"),
    ),
    ui.nav("Wielkości odstające",
           ui.row(
               ui.column(6,
                         ui.input_select("method", "Wybierz kryterium", ["Q +/- 1.5IQR", "Q +/- 3IQR"])
                         ),
                ui.column(6, 
                          ui.input_select("numeric_outliers", "Wybierz zmienną", [])
                          ),
                ),
            ui.output_data_frame("outliers_df"),
           ),

    title = "Analiza Danych",
    bg = "#09171c",
)


def server(input, output, session):

    @reactive.Calc
    @reactive.event(input.ready_data, input.file1, ignore_none=True)
    def parsed_file():

        if input.ready_data():
            path = file_path
            df = pd.read_csv(path, index_col=False) 
            return df
        
        file: list[FileInfo] | None = input.file1()
        
        if file is None:
            return pd.DataFrame()
            
        df = pd.read_csv(file[0]["datapath"], index_col=False)
        
        return df

    
    @reactive.Calc
    def choose_categories():
        data = parsed_file()
        data.dropna(axis=1, how='all', inplace=True)
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns

        return list(categorical_vars)
    
    @reactive.Calc
    def choose_numeric():
        data = parsed_file()
        data.dropna(axis=1, how='all', inplace=True)
        numeric_vars = data.select_dtypes(include=['int64', 'float64']).columns

        return list(numeric_vars)
    
    @reactive.Calc
    def stat_agg():

        def q1(x):
            return x.quantile(0.25)

        def q3(x):
            return x.quantile(0.75)

        function_map = {
            'q1': q1,
            'q3': q3
        }

        inp = input.statistic()
        stat = function_map[inp]

        return stat

    @reactive.Effect
    def _():
        x = choose_categories()

        if x is None:
            x = []
        elif isinstance(x, str):
            x = [x]

        ui.update_select(
        "category_stat",
        
        choices = [f"{str(i)}" for i in x],
        )

    @reactive.Effect
    def _():
        x = choose_categories()

        if x is None:
            x = []
        elif isinstance(x, str):
            x = [x]

        ui.update_select(
        "category_bar",
        
        choices = [f"{str(i)}" for i in x],
        )
    
    @reactive.Effect
    def _():
        x = choose_categories()

        if x is None:
            x = []
        elif isinstance(x, str):
            x = [x]

        ui.update_select(
        "category_time",
        
        choices = [f"{str(i)}" for i in x],
        )

    @reactive.Effect
    def _():
        x = choose_categories()

        if x is None:
            x = []
        elif isinstance(x, str):
            x = [x]

        ui.update_select(
        "category_agg",
        
        choices = [f"{str(i)}" for i in x],
        )
    
    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_selectize(
        "numeric_stat", choices = [f"{str(i)}" for i in y],
        )

    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_select(
        "numeric_hist",
        
        choices = [f"{str(i)}" for i in y],
        )

    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_select(
        "numeric_corr", choices = [f"{str(i)}" for i in y],
        )

    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_select(
        "numeric_corr2", choices = [f"{str(i)}" for i in y],
        )

    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_selectize(
        "numeric_box", choices = [f"{str(i)}" for i in y],
        )


    @reactive.Effect
    def _():
        z = choose_numeric()
        data = parsed_file()

        if z is None:
            z = []
        elif isinstance(z, str):
            z = [z]  

        if "Szereg_czasowy" in data.columns:
            ui.update_select(
            "time", choices = [f"{str(i)}" for i in z],
            selected="Szereg_czasowy"
        )
        else:
            ui.update_select(
                "time", choices = [f"{str(i)}" for i in z]
            )

    @reactive.Effect
    def _():
        z = choose_numeric()

        if z is None:
            z = []
        elif isinstance(z, str):
            z = [z]  
            
        ui.update_select(
            "variable", choices = [f"{str(i)}" for i in z]
        )

    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_select(
        "numeric_agg", choices = [f"{str(i)}" for i in y],
        )

    @reactive.Effect
    def _():
        y = choose_numeric()

        if y is None:
            y = []
        elif isinstance(y, str):
            y = [y]

        ui.update_select(
        "numeric_outliers", choices = [f"{str(i)}" for i in y],
        )
  
    @output
    @render.data_frame
    def summary_data():
        data = parsed_file()
        
        return render.DataGrid(
            data.head(int(input.number_of_rows())),
            height=350,
            width="100%",
            filters=True,
        )
    
    @render.data_frame
    def charakterystyka_zbioru():
        data = parsed_file()
        df_dim = data.shape

        data_info = pd.DataFrame(
            {
                "Zmienna": [
                    "Liczba wierszy",
                    "Liczba kolumn",
                    "Liczba kolumn numerycznych",
                    "Liczba kolumn symbolicznych",
                ],
                "Wartość": [
                    df_dim[0],
                    df_dim[1],
                    data.select_dtypes(include="number").shape[1],
                    data.select_dtypes(include="object").shape[1],
                ],
            }
        )

        return render.DataGrid(data_info, width="100%", height="100%")

    @render.data_frame
    def wartosci_nan_ch():
        data = parsed_file()
        tmp = ((len(data) - data.count()) / len(data)) * 100
        missing_data = pd.DataFrame({"Zmienna": tmp.index, "Wartość [%]": tmp.values})

        return render.DataGrid(missing_data, width="100%", height="100%")

    @render.data_frame
    def num_stats():
        data = parsed_file()
        x = input.numeric_stat()
        req(x)

        stat_num = pd.DataFrame()
        for i in range(len(x)):
            tmp = data[x[i]].describe()
            stat_num.insert(i, x[i], np.around(tmp.values))
        
        stat_num.insert(0, "Statystyki", tmp.index)


        return render.DataGrid(stat_num, width="100%", height="100%")

    @render.data_frame
    def atr_stats():
        data = parsed_file()
        x = choose_categories()
        y = input.category_stat()
        req(y)

        if len(x) == 0:
            return pd.DataFrame()
        
        stat_atr = pd.DataFrame()
        for i in range(len(y)):
            tmp = data[y[i]].describe()
            stat_atr.insert(i, y[i], tmp.values)
        
        stat_atr.insert(0, "Statystyki", tmp.index)


        return render.DataGrid(stat_atr, width="100%", height="100%")

    @render_widget
    def histogram():
        data = parsed_file()
        data.dropna(axis=1, how='all', inplace=True)

        fig = px.histogram(
            data,
            x=input.numeric_hist(),
            nbins=input.bins(),
            template="plotly_white",
            color_discrete_sequence=["#df6919"]
        )
        return fig

    @render_widget
    def correlation():
        data = parsed_file()
        fig = px.scatter(
            data, x=input.numeric_corr(), 
            y=input.numeric_corr2(), 
            template="plotly_white",
            color_discrete_sequence=["#df6919"])

        return fig

    @render_widget
    def boxplot():
        data = parsed_file()
        
        fig = px.box(
            data, y = list(input.numeric_box()),
            template="plotly_white",
            color_discrete_sequence=["#df6919"])
        
        return fig

    @render_widget
    def barplot_atr():
        data = parsed_file()

        x = choose_categories()
        
        if len(x) == 0:
            return 

        df_tmp = data[input.category_bar()].value_counts()
        df_count = pd.DataFrame(
            {"Kategoria": df_tmp.index, "count": df_tmp.values}
        )

        fig = px.bar(
            df_count,
            x="Kategoria",
            y="count",
            title="Ilość wystąpienia danej kategorii",
            template="plotly_white",
            color_discrete_sequence=["#df6919"],
            height=500
        )
        return fig

    @render.data_frame
    def corr_df():
        data = parsed_file()
        x = input.numeric_corr()
        y = input.numeric_corr2()

        if len(x) == 0 or len(y) == 0:
            return pd.DataFrame()


        correlation = data[x].corr(data[y], method=input.method_corr())
        corr_df = pd.DataFrame({"Korelacja": [correlation], "Metoda": [input.method_corr()]})
        
        return render.DataGrid(corr_df, width="100%", height="100%")
    
    @render_widget
    def time_city():
        data = parsed_file()

        cat = input.category_time()

        df = data.sort_values(input.time())

        if cat is None:
            df_tmp = df.groupby(input.time())[input.variable()].mean()
            fig = px.line(
            df_tmp,
            x=df_tmp.index,
            y=df_tmp.values,
            color=input.category_time(),
            markers=True,
            template="plotly_white",
        )
        else:
            fig = px.line(
            df,
            x=input.time(),
            y=input.variable(),
            color=input.category_time(),
            markers=True,
            template="plotly_white",
        )

        return fig

    
    @render.data_frame
    def grouped_data():


        data = parsed_file()
        stat = []
        x = choose_categories()
        req(input.statistic())
        stat.append(input.statistic())
       
        if len(stat) == 0 or len(x) == 0:
            return pd.DataFrame()
        
        df = data.groupby(input.category_agg())[input.numeric_agg()].agg(input.statistic())
        df.insert(0,"Kategoria", df.index)
        return render.DataGrid(df, width="100%", height="100%")
    
    @render.data_frame
    def outliers_df():
        data = parsed_file()
        x = input.numeric_outliers()

        if input.method() == "Q +/- 1.5IQR":
            cryt = 1.5
        elif input.method() == "Q +/- 3IQR":
            cryt = 3.0

        m = np.median(data[x])
        q1 = np.quantile(data[x], 0.25)
        q3 = np.quantile(data[x], 0.75)
        iqr = q3 - q1

        indices = [i for i, num in enumerate(data[x]) if num < q1 - cryt * iqr or num > q3 + cryt * iqr]

        if len(indices) == 0:
            df = pd.DataFrame({" ": ["Brak wartości odstających"]})
            return render.DataGrid(df, width="100%", height="100%")
        else:
            outliers = data.iloc[indices]
            return render.DataGrid(outliers, width="100%", height="100%")
       

app = App(app_ui, server)