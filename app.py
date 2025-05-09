import shinyswatch
from shiny import App, ui, reactive, render, Inputs, Outputs, Session
from pathlib import Path
from faicons import icon_svg

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64
from fpdf import FPDF
from PIL import Image
from zipfile import ZipFile

from numpy import random
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from datetime import datetime
import time
from scipy.special import ndtri

import cProfile
import pstats
import csv
import sys

from utils import quantile_search, optimal_shift, generate_pdf_from_base64_images, plot_loss_distribution, AC_fit, Bivarcumnorm, fast_weighted_avg, fast_vector_mult, fast_norm_cdf
from utils import Migration_events, ensure_numpy, format_eta

www_dir = Path(__file__).parent / "www"

# Upload mapping tables for stochastic LGD modelling
data = np.load(www_dir / "MappingTables.npz")
RhoMapping = data['table2']
LGD_cond_var_Mapping = data['table1']

#os.chdir("C:/Users/CarstenGerken/OneDrive - True North Partners LLP/CPRAxx/CPRAxx - Python")

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app_ui = ui.page_fluid(

    ui.tags.link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Bruno+Ace+SC&display=swap"
    ),

    ui.tags.style(
        "h1 { font-family: 'Bruno Ace SC', sans-serif; }",
        ".card-header { color:white; background: #2D89C8; }"
    ),

    
    ui.layout_columns(
        ui.div(
            ui.img(src='CPRAxx_Logo_New.png', width="191.5"),
            style="display: flex; align-items: center; justify-content: center; height: 150px;"
        ),
        ui.div(
            ui.h1("Credit Portfolio Risk Analyser", 
                style="color:#2D89C8; font-style: italic; font-weight: bold; text-align: center; display: flex; align-items: center; justify-content: center; height: 100%;"
            ),
            style="display: flex; align-items: center; justify-content: center; height: 150px;"
        ),
        ui.div(
            ui.img(src='tnpartners-logo.jpg'),
            style="display: flex; align-items: center; justify-content: center; height: 150px;"
        ),
        col_widths=(2, 7, 3)
    ),
    
    # Logout, Run, and Close Buttons
    ui.layout_columns(
        None,
        ui.input_action_button("logout", "User log out", class_="btn-danger",
                               style="color: #fff; background-color: #2D89C8; border-color: #2e6da4; font-size: 9pt"
        ),
        ui.input_action_button(
            "run",
            "Kick off run", 
            icon=icon_svg("circle-play"),
            class_="btn-primary", 
            style="color: #fff; background-color: #2D89C8; border-color: #2e6da4; font-size: 9pt", 
            width=110
        ),
        ui.input_action_button(
            "close",
            "Close app",
            icon=icon_svg("right-from-bracket"),
            class_="btn-primary",
            style="color: #fff; background-color: #2D89C8; border-color: #2e6da4; font-size: 9pt" 
        ),
        col_widths=(9, 1, 1, 1)
    ),
    
     # Navigation Panel
    ui.card(
        ui.card_header(ui.h4("Input & Settings")),
        ui.navset_pill_list(
            ui.nav_panel("Model run types",
                ui.panel_well(
                    ui.p("Selection of model run type", style = "font-style: italic; font-weight: bold"),
                    ui.input_radio_buttons(
                        "RunType", "", {1:"Capital window", 2:"Allocation", 3:"Stand-alone pricing"}
                    ),
                )
            ),
            ui.nav_panel("Simulation settings",
                ui.panel_conditional(
                    "input.RunType == 3",
                    ui.panel_well(
                        ui.p("Simulation settings cannot be changed for stand-alone pricing run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                ),
                ui.panel_conditional(
                    "input.RunType <= 2",          
                    ui.panel_well(
                        ui.p("Correlation input type", style = "font-style: italic; font-weight: bold"),
                        ui.input_radio_buttons(
                            "Cor_Input_Type", "", {1:"Correlation matrix", 2:"Factor table"}
                        ),
                    ),
                    ui.p(""),
                    ui.panel_well(
                        ui.row(
                            ui.column(6,
                                ui.p("Number of simulations", style = "font-style: italic; font-weight: bold"),
                                ui.input_numeric("NumberOfSimulations", "", value=10000),
                                ui.p("Number of seeds", style = "font-style: italic; font-weight: bold"),
                                ui.input_numeric("NumberOfSeeds", "", value=4),
                            ),
                            ui.column(6,
                                ui.p("Maximum pool size for individual treatment", style = "font-style: italic; font-weight: bold"),
                                ui.input_numeric("MaxPoolSize", "", value=10)
                            ),
                        )
                    ),
                    ui.p(""),
                    ui.panel_well(
                        ui.p("Number of confidence levels", style = "font-style: italic; font-weight: bold"),
                        ui.input_slider(
                            "NumberOfCIs", "", min=1, max=5, value=3
                        ),
                        ui.row(
                            ui.column(6,
                                ui.p("Confidence levels (in %)", style = "font-style: italic; font-weight: bold"),
                                ui.input_numeric("Quantiles1", "", value=99.93),
                            ),
                            ui.column(6,
                                ui.panel_conditional(
                                    "input.RunType == 2",
                                    ui.p("of which are used for capital allocation", style = "font-style: italic; font-weight: bold"),
                                    ui.input_checkbox("AllocationQuantile1", "")
                                )
                            ),
                        ),
                        ui.panel_conditional(
                            "input.NumberOfCIs>=2",
                            ui.row(
                                ui.column(6,
                                        ui.input_numeric("Quantiles2", "", value=99.9)
                                ),
                                ui.column(6,
                                    ui.panel_conditional(
                                        "input.RunType == 2",
                                        ui.input_checkbox("AllocationQuantile2", "")
                                    )
                                )
                            ),
                        ),
                        ui.panel_conditional(
                            "input.NumberOfCIs>=3",
                            ui.row(
                                ui.column(6,
                                        ui.input_numeric("Quantiles3", "", value=99)
                                ),
                                ui.column(6,
                                    ui.panel_conditional(
                                        "input.RunType == 2",
                                        ui.input_checkbox("AllocationQuantile3", "")
                                    )
                                )
                            ),
                        ),
                        ui.panel_conditional(
                            "input.NumberOfCIs>=4",
                            ui.row(
                                ui.column(6,
                                        ui.input_numeric("Quantiles4", "", value=95)
                                ),
                                ui.column(6,
                                    ui.panel_conditional(
                                        "input.RunType == 2",
                                        ui.input_checkbox("AllocationQuantile4", "")
                                    )
                                )
                            ),
                        ),
                        ui.panel_conditional(
                            "input.NumberOfCIs>=5",
                            ui.row(
                                ui.column(6,
                                        ui.input_numeric("Quantiles5", "", value=90)
                                ),
                                ui.column(6,
                                    ui.panel_conditional(
                                        "input.RunType == 2",
                                        ui.input_checkbox("AllocationQuantile5", "")
                                    )
                                )
                            )
                        )
                    ),
                    ui.p(""),
                    ui.panel_well(
                        ui.row(
                            ui.column(6,
                                ui.input_checkbox("MarketValueApproach", "Include rating migrations")
                            ),
                            ui.column(6,
                                ui.input_numeric("Rand_seed",ui.h6("Random seed selection",style = "font-style: italic"),value = 0)
                            )
                        )
                    )
                )
            ),
            ui.nav_panel("Simulation settings - LGD",
                ui.panel_conditional(
                    "input.RunType == 3",
                    ui.panel_well(
                        ui.p("LGD simulation settings cannot be changed for stand-alone pricing run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                ),
                ui.panel_conditional(
                    "input.RunType <= 2",             
                    ui.panel_well(
                        ui.p("LGD simulation approach ", style = "font-style: italic; font-weight: bold"),
                        ui.input_radio_buttons(
                            "LGD_Sim_Type", "", {1: "Deterministic LGDs", 2:"Stochastic LGDs w/o PD/LGD correlations", 3:"Stochastic LGDs with PD/LGD correlations"}
                        )
                    ),
                    ui.panel_conditional(
                        "input.LGD_Sim_Type >= 2",
                        ui.p(""),             
                        ui.panel_well(
                            ui.row(
                                ui.input_checkbox("LGD_Pool","Apply pooled treatment only if number of defaults exeeds:"),
                                ui.input_numeric("LGD_Pool_min", "", value=10)
                                
                            )
                        )
                    )
                )
            ),
            ui.nav_panel("Input data",
                        ui.panel_well(
                            ui.p("Portfolio input data", style = "font-style: italic; font-weight: bold"),
                            ui.input_file("file1", "", accept=[".csv"]),
                            ui.panel_conditional(
                                "input.Cor_Input_Type == 1",  
                                ui.p("Correlation matrix", style = "font-style: italic; font-weight: bold"),
                                ui.input_file("file2", "", accept=[".csv"]),
                            ),
                            ui.panel_conditional(
                                "input.Cor_Input_Type == 2",  
                                ui.p("Factor table", style = "font-style: italic; font-weight: bold"),
                                ui.input_file("file2_factor", "", accept=[".csv"]),
                            ),
                            ui.panel_conditional(
                                "input.LGD_Sim_Type == 2",  
                                ui.p("LGD correlation matrix", style = "font-style: italic; font-weight: bold"),
                                ui.input_file("file3", "", accept=[".csv"])
                            ),
                            ui.panel_conditional(
                                "input.MarketValueApproach == true",  
                                ui.p("Migration matrix", style = "font-style: italic; font-weight: bold"),
                                ui.input_file("file4", "", accept=[".csv"])
                            ),
                        )
            ),
            ui.nav_panel("Capital allocation settings",
                ui.panel_conditional(
                    "input.RunType == 1",
                    ui.panel_well(
                        ui.p("Capital allocation settings are not relevant for capital window run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                ),
                ui.panel_conditional(
                    "input.RunType == 2",
                    ui.panel_well(
                        ui.input_checkbox("EScontrib","Expected shortfall contribution", value = True),
                        ui.panel_conditional(
                            "input.EScontrib",
                            ui.row(
                                ui.column(1,None),
                                ui.column(5,
                                    ui.input_checkbox("EScontrib_sim","Based on actual simulations", value = True),
                                    ui.input_checkbox("EScontrib_analytic","Based on deterministic approximation", value = False),
                                    ui.input_radio_buttons(
                                        "ESType",ui.h6("Allocation approach",style = "font-style: italic"),
                                        {1:"VaR/ES match", 2:"VaR allocation", 3:"Application of lower boundary"},
                                        selected=1
                                    ),
                                    ui.panel_conditional(
                                        "input.ESType == 3",
                                        ui.input_numeric("LowerBoundary_ES",ui.h6("Lower boundary",style = "font-style: italic"),value = 0)
                                    )
                                ),
                                ui.column(6,None)
                            )
                        )
                    ),
                    ui.p(""),
                    ui.panel_well(
                        ui.input_checkbox("MRcontribSim","Marginal risk contribution (simulated)", value = False),
                    ),
                    ui.p(""),
                    ui.panel_well(
                        ui.input_checkbox("WCE","Window conditional expectation",value=False),
                        ui.panel_conditional(
                            "input.WCE",
                            ui.row(
                                ui.column(1,None),
                                ui.column(8,
                                    ui.input_checkbox("WCE_sim","Based on actual simulations", value = True),
                                    ui.input_checkbox("WCE_analytic","Based on deterministic approximation", value = False),
                                    ui.input_slider(
                                        "NumberOfWindows", ui.h6("Number of windows",style = "font-style: italic"), min=1, max=5, value=3
                                    )
                                ),
                                ui.column(3,None)
                            ),
                            ui.row(
                                ui.column(1,None),
                                ui.column(4,
                                    ui.h6("Upper confidence levels (in %)", style = "font-style: italic"),
                                    ui.input_numeric("UpperBoundary1", "", value=99.98)
                                ),
                                 ui.column(4,
                                    ui.h6("Lower confidence levels (in %)", style = "font-style: italic"),
                                    ui.input_numeric("LowerBoundary1", "", value=99.88)
                                ),
                                ui.column(3,None)
                            ),
                            ui.panel_conditional(
                                "input.NumberOfWindows>=2",
                                ui.row(
                                    ui.column(1,None),
                                    ui.column(4,
                                        ui.input_numeric("UpperBoundary2", "", value=99.95)
                                    ),
                                    ui.column(4,
                                       ui.input_numeric("LowerBoundary2", "", value=99.85)
                                    ),
                                    ui.column(3,None)
                                )
                            ),
                            ui.panel_conditional(
                                "input.NumberOfWindows>=3",
                                ui.row(
                                    ui.column(1,None),
                                    ui.column(4,
                                        ui.input_numeric("UpperBoundary3", "", value=99.5)
                                    ),
                                    ui.column(4,
                                       ui.input_numeric("LowerBoundary3", "", value=98.5)
                                    ),
                                    ui.column(3,None)
                                )
                            ),
                            ui.panel_conditional(
                                "input.NumberOfWindows>=4",
                                ui.row(
                                    ui.column(1,None),
                                    ui.column(4,
                                        ui.input_numeric("UpperBoundary4", "", value=97.5)
                                    ),
                                    ui.column(4,
                                       ui.input_numeric("LowerBoundary4", "", value=92.5)
                                    ),
                                    ui.column(3,None)
                                )
                            ),
                            ui.panel_conditional(
                                "input.NumberOfWindows>=5",
                                ui.row(
                                    ui.column(1,None),
                                    ui.column(4,
                                        ui.input_numeric("UpperBoundary5", "", value=95)
                                    ),
                                    ui.column(4,
                                       ui.input_numeric("LowerBoundary5", "", value=85)
                                    ),
                                    ui.column(3,None)
                                )
                            )
                        )
                    )
                ),
                 ui.panel_conditional(
                    "input.RunType == 3",
                    ui.panel_well(
                        ui.p("Capital allocation settings cannot be changed for stand-alone pricing run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                )
            ),
            ui.nav_panel("Importance sampling settings",
                ui.panel_conditional(
                    "input.RunType <= 2",
                    ui.panel_well(
                        ui.input_checkbox("ImportanceSampling","Apply importance sampling", value = True),
                        ui.panel_conditional(
                            "input.ImportanceSampling",
                            ui.row(
                                ui.column(5,
                                    ui.input_select("ISType", ui.h6("Importance sampling method",style = "font-style: italic"),
                                                    {1:"Volatility scaling (Morokoff)", 2:"Mean shift (Kalkbrener, all factors)", 3:"Mean shift (Kalkbrener, only first factor)"}, selected = 1
                                    )
                                ),
                                ui.column(3,
                                    ui.panel_conditional(
                                        "input.ISType == 1",
                                        ui.input_numeric("ScalingFactor",ui.h6("Parameter settings",style = "font-style: italic"),value = 2.5)
                                    ),
                                    ui.panel_conditional(
                                        "input.ISType >= 2",
                                        ui.input_numeric("AmplificationFactor",ui.h6("Parameter settings",style = "font-style: italic"),value = 1.0)
                                    )
                                ),
                                ui.column(4,None)
                            )
                        ),
                        ui.input_checkbox("AntitheticSampling","Apply antithetic sampling", value = True)
                    )
                ),
                 ui.panel_conditional(
                    "input.RunType == 3",
                    ui.panel_well(
                        ui.p("Importance sampling settings cannot be changed for stand-alone pricing run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                 ),
            ),
            ui.nav_panel("Parallel computing settings",
                ui.panel_conditional(
                    "input.RunType <= 2",
                    ui.panel_well(
                        ui.input_checkbox("ParallelComp","Activate parallel computing", value = False),
                        ui.panel_conditional(
                            "input.ParallelComp",
                            ui.row(
                                ui.column(1,None),
                                ui.column(5,
                                    ui.input_numeric("NumberOfCores",ui.h6("Number of cores",style = "font-style: italic"),value = 2)
                                )
                            ),
                            ui.row(
                                ui.column(1,None),
                                ui.column(5,
                                    ui.input_radio_buttons(
                                        "ParallelType",ui.h6("Parallel computing type",style = "font-style: italic"),
                                        {1:"by seed", 2:"by seed and simulation step (batched)", 3:"by simulation step (batched)"},
                                        selected=1
                                    )
                                ),
                                ui.column(3,
                                    ui.panel_conditional(
                                        "input.ParallelType==2",
                                        ui.input_numeric("BatchSize_Sim1",ui.h6("Number of batches",style = "font-style: italic"),value = 2)
                                    ),
                                    ui.panel_conditional(
                                        "input.ParallelType==3",
                                        ui.input_numeric("BatchSize_Sim",ui.h6("Number of batches",style = "font-style: italic"),value = 2)
                                    )
                                )
                            )
                        )
                    )
                ),
                ui.panel_conditional(
                    "input.RunType == 3",
                    ui.panel_well(
                        ui.p("Parallel computing settings cannot be changed for stand-alone pricing run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                 ),
            ),
            ui.nav_panel("Output and other settings",
                ui.panel_well(
                    ui.p("Output settings", style = "font-style: italic; font-weight: bold"),
                    ui.input_checkbox("GraphicalOutput","Graphical output", value = False),
                    ui.input_checkbox("LossDistribution2","Loss distribution output (full)", value = False),
                    ui.input_checkbox("LossDistribution","Loss distribution output (quantiles only)", value = False),
                    ui.panel_conditional(
                        "input.RunType >= 2",
                        ui.input_checkbox("CondPDLGD","Output of conditional default and loss rates", value = False)
                    ),
                    ui.input_checkbox("PricingOutput","Store data for pricing runs", value = False),
                    ui.panel_conditional(
                        "input.LGD_Sim_Type == 1",
                        ui.input_checkbox("AnalyticalProxies","Comparison to analytical approximations", value = False)
                    )
                ),
                ui.p(""),
                ui.panel_well(
                    ui.p("Other settings", style = "font-style: italic; font-weight: bold"),
                    ui.panel_conditional(
                        "input.RunType == 2",
                        ui.input_checkbox("CapitalGrid","Include homogeneous capital grid to measure concentration risk", value = False)
                    )
                )
            ),
            ui.nav_panel("Pricing settings",
                ui.panel_well(
                    ui.panel_conditional(
                        "input.RunType == 1",
                        ui.panel_well(
                            ui.p("Pricing is not possible in a capital window run", style = "font-style: italic; font-weight: bold; color: red")
                        )
                    ),
                    ui.panel_conditional(
                        "input.RunType == 2",
                        ui.input_radio_buttons(
                            "PricingType","",
                            {1:"No pricing", 2:"Pricing of new deal (full model run)"},
                            selected=1
                        )
                    ),
                    ui.panel_conditional(
                        "input.RunType == 3",
                        ui.input_radio_buttons(
                            "PricingType2","",
                            {1:"Pricing of new deal (stand-alone analysis)"},
                            selected=1
                        )
                    ),
                    ui.panel_conditional(
                        "(input.RunType == 2 && input.PricingType == 2) || input.RunType == 3",
                        ui.p(""),
                        ui.panel_well(
                            ui.row(
                                ui.column(2,
                                    ui.input_text("ExposureID_Pricing",ui.h6("Exposure ID",style = "font-style: italic"),value="")
                                ),
                                ui.column(2,
                                    ui.input_text("GroupID_Pricing",ui.h6("Group ID",style = "font-style: italic"),value="")
                                ),
                                ui.column(2,
                                    ui.input_numeric("EAD_Pricing",ui.h6("EAD",style = "font-style: italic"),value=1000000)
                                ),
                                ui.column(2,
                                    ui.input_numeric("LGD_Pricing",ui.h6("LGD",style = "font-style: italic"),value=0.5)
                                ),
                                ui.column(3,
                                    ui.input_numeric("PD_Pricing",ui.h6("PD",style = "font-style: italic"),value=0.01)
                                )
                            ),
                            ui.row(
                                ui.column(2,
                                    ui.input_text("Segment_Pricing",ui.h6("Segment",style = "font-style: italic"),value="Sovereign")
                                ),
                                ui.column(2,
                                    ui.input_numeric("Rsquared_Pricing",ui.h6("R-squared",style = "font-style: italic"),value=0.3)
                                ),
                                ui.column(2,
                                    ui.input_text("LGD_Segment_Pricing",ui.h6("LGD segment",style = "font-style: italic"),value="Sovereign")
                                ),
                                ui.column(2,
                                    ui.input_numeric("k_Parameter_Pricing",ui.h6("k parameter",style = "font-style: italic"),value=4)
                                ),
                                ui.column(3,
                                    ui.input_numeric("LGD_Rsquared_Pricing",ui.h6("LGD R-squared",style = "font-style: italic"),value=0.2)
                                )
                            ),
                            ui.panel_conditional(
                                "input.MarketValueApproach == true",
                                ui.p("Additional settings for market value approach", style = "font-style: italic; font-weight: bold"),
                                ui.row(
                                    ui.column(2,
                                        ui.input_text("RatingClass_Pricing",ui.h6("Rating class",style = "font-style: italic"),value="AA")
                                    ),
                                    ui.column(3,
                                        ui.input_numeric("TimeToMaturity_Pricing",ui.h6("Time to maturity (in yrs)",style = "font-style: italic"),value=3.0)
                                    ),
                                    ui.column(3,
                                        ui.input_numeric("Yield_Pricing",ui.h6("Effective interest rate",style = "font-style: italic"),value=0.03)
                                    ),
                                    ui.column(3,
                                        ui.input_select("Approach_Pricing",ui.h6("Approach",style = "font-style: italic"),
                                                        {1:"Default", 2:"Market value"}, selected=1)
                                    )
                                )
                            )
                        ),
                        ui.panel_conditional(
                            "input.RunType == 3",
                            ui.p(""),
                            ui.panel_well(
                                ui.p("Pricing input (output file from initial run)", style = "font-style: italic; font-weight: bold"),
                                ui.row(
                                    ui.column(11,
                                        ui.input_file("file6", "", accept=[".csv"])
                                    )
                                ) 
                            )
                        )
                    )
                )
            ),

     
            id="tab_Input",
            widths=(2, 6)        
        )
    ), 

    ui.card(
        ui.card_header(ui.h4("Output & Results")),
        ui.navset_pill_list(
            ui.nav_panel("Results summary",
                ui.panel_well(
                    ui.output_ui("PortfolioResults_summary")
                ),
                ui.panel_conditional(
                    "input.MarketValueApproach == true",
                    ui.HTML("<br>"),
                    ui.panel_well(
                        ui.p("Standalone results for default events", style = "font-style: italic; font-weight: bold"),
                        ui.output_ui("PortfolioResults_summary_default")
                    ),
                    ui.HTML("<br>"),
                    ui.panel_well(
                        ui.p("Standalone results for migration events (excl. default)", style = "font-style: italic; font-weight: bold"),
                        ui.output_ui("PortfolioResults_summary_migration")
                    ),
                ),
            ),
            ui.nav_panel("Loss distributions - Graphical",
                ui.panel_well(
                    ui.output_ui("histograms")
                )
            ),
            ui.nav_panel("Loss distributions - Tabular",
                ui.panel_well(
                    ui.output_ui("LossDistribution_quantiles")
                )
            ),
            ui.nav_panel("Allocation details",
                ui.panel_conditional(
                    "input.RunType == 1",
                    ui.panel_well(
                        ui.p("Allocation details are not available for a capital window run", style = "font-style: italic; font-weight: bold; color: red")
                    )
                ),
                ui.panel_conditional(
                    "input.RunType >= 2",
                    ui.panel_well(
                        ui.output_ui("Allocation_summary")
                    )
                )
            ),
            ui.nav_panel("Allocation statistics",
                ui.row(
                    ui.column(9,
                        ui.panel_conditional(
                            "input.RunType == 1",
                            ui.panel_well(
                                ui.p("Allocation statistics are not available for a capital window run", style = "font-style: italic; font-weight: bold; color: red")
                            )
                        ),
                        ui.panel_conditional(
                            "input.RunType >= 2 && input.EScontrib == true",
                            ui.panel_well(
                                ui.p("Number of scenarios for expected shortfall contribution", style = "font-style: italic; font-weight: bold"),
                                ui.output_ui("EScontrib_summary")
                            )
                        ),
                        ui.panel_conditional(
                            "input.RunType >= 2 && input.EScontrib == true && input.WCE == true",
                            ui.HTML("<br>")
                        ),
                        ui.panel_conditional(
                            "input.RunType >= 2 && input.EScontrib == true && input.WCE == false && input.NumberOfSeeds > 1",
                            ui.HTML("<br>")
                        ),
                        ui.panel_conditional(
                            "input.RunType >= 2 && input.WCE == true",
                            ui.panel_well(
                                ui.p("Number of scenarios for window conditional expectation", style = "font-style: italic; font-weight: bold"),
                                ui.output_ui("WCE_summary")
                            )
                        ),
                        ui.panel_conditional(
                            "input.RunType >= 2 && input.WCE == true && input.NumberOfSeeds > 1",
                            ui.HTML("<br>")
                        ),
                        ui.panel_conditional(
                            "input.RunType >= 2 && input.NumberOfSeeds > 1",
                            ui.panel_well(
                                ui.p("Stability of capital allocation (coefficient of variation, in %)", style = "font-style: italic; font-weight: bold"),
                                ui.output_ui("Stability_summary")
                            )
                        )
                    ),
                    ui.column(3,"")
                )
            ),
            ui.nav_panel("Download of results",
                ui.row(
                    ui.column(9,
                        ui.panel_well(
                            ui.panel_well(
                                ui.p("Download of results", style = "font-style: italic; font-weight: bold"),
                                ui.download_button("downloadData", "Full results package (zip)",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; ")
                            ),
                            ui.p(""),
                            ui.panel_well(
                                ui.p("Download of individual files", style = "font-style: italic; font-weight: bold"),
                                ui.row(
                                    ui.column(6,
                                        ui.download_button("downloadPortfolioResults", "Portfolio level results",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ui.p(""),
                                        ui.download_button("downloadAllocationResults", "Allocation results",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ui.p(""),
                                        ui.download_button("downloadAllocationStability", "Allocation stability",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ui.panel_conditional(
                                            "input.MarketValueApproach == true",
                                            ui.p(""),
                                            ui.download_button("downloadPortfolioResults_default", "Portfolio level results (defaults)",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ),
                                        ui.p(""),
                                        ui.download_button("downloadSettings", "Run settings",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                    ),
                                    ui.column(6,
                                        ui.download_button("downloadLossDist_quantiles", "Loss distribution (quantiles only)",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ui.p(""),
                                        ui.download_button("downloadLossDist_full", "Loss distribution (full)",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ui.p(""),
                                        ui.download_button("downloadLossDist_graph", "Loss distribution (pdf plot)",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                         ui.panel_conditional(
                                            "input.MarketValueApproach == true",
                                            ui.p(""),
                                            ui.download_button("downloadPortfolioResults_migration", "Portfolio level results (migrations)",width="10cm",icon=icon_svg("file-export"), style="text-align: left; display: inline-block; "),
                                        ), 
                                    )
                                )
                            )
                        )
                    ),
                    ui.column(3,"")
                )
            ),
            ui.nav_panel("Summary of settings",
                ui.row(
                    ui.column(9,
                        ui.panel_well(
                            ui.p("Summary of run settings", style = "font-style: italic; font-weight: bold"),
                            ui.output_ui("Settings_summary")
                        )
                    ),
                    ui.column(3,"")
                )
            ),
            id="tab_Output",
            widths=(2, 8)        
        )
    ), 




    # Platz für zukünftige UI-Elemente
    ui.output_plot("p")
)


def server(input: Inputs, output: Outputs, session: Session):
    
    histogram_images = reactive.Value([])
 
    @output
    @render.ui
    def dynamic_tabs_capital_allocation():
        if input.RunType() == 2:
            return ui.nav_panel("Capital allocation settings",
                    ui.p("Migration matrix", style = "font-style: italic; font-weight: bold")
            ),
        return ui.nav_panel("") 
        
    @reactive.effect
    @reactive.event(input.close)
    def close_app():
        # Inject JavaScript to close the window
        ui.insert_ui(
            ui.HTML("<script>setTimeout(function(){ window.close(); }, 500);</script>"),
            selector="body"
        )

    
    @reactive.effect
    @reactive.event(input.run, ignore_none=True)
    def p():

        StartDate=datetime.now()
        
        ##########################################
        ### Coversion of inputs into variables ###
        ##########################################

        RunType1=False
        RunType2=False
        RunType3=False
    
        if input.RunType()=="1":
            RunType1=True
        elif input.RunType()=="2":
            RunType2=True
        elif input.RunType()=="3":
            RunType3=True
        
        Deterministic=False
        Vasicek=False
        PLC=False
  
        if input.LGD_Sim_Type()=="1":
            Deterministic=True
        elif input.LGD_Sim_Type()=="2":
            Vasicek=True
        elif input.LGD_Sim_Type()=="3":
            PLC=True
        
        ISType1=False
        ISType2=False
        ISType3=False
        
        if input.ISType()=="1":
            ISType1=True
        elif input.ISType()=="2":
            ISType2=True
        elif input.ISType()=="3":
            ISType2=True
            ISType3=True

        ParallelType1=False
        ParallelType3=False
        ParallelType4=False
           
        if input.ParallelType()=="1":
            ParallelType1=True
        elif input.ParallelType()=="2":
            ParallelType3=True
        elif input.ParallelType()=="3":
            ParallelType4=True
        
        PricingType0=False
        PricingType1=False
        PricingType2=False

        if input.PricingType()=="1" or RunType1:
            PricingType0=True
        else:
            if input.PricingType()=="2" and RunType2:
                PricingType1=True
            if RunType3:
                PricingType2=True

        if PricingType1 or PricingType2:
            ExposureID_Pricing=input.ExposureID_Pricing()
            GroupID_Pricing=input.GroupID_Pricing()
            EAD_Pricing=input.EAD_Pricing()
            LGD_Pricing=input.LGD_Pricing()
            PD_Pricing=input.PD_Pricing()
            Segment_Pricing=input.Segment_Pricing()
            Rsquared_Pricing=input.Rsquared_Pricing()
            LGD_Segment_Pricing=input.LGD_Segment_Pricing()
            k_Parameter_Pricing=input.k_Parameter_Pricing()
            LGD_Rsquared_Pricing=input.LGD_Rsquared_Pricing()
            RatingClass_Pricing=input.RatingClass_Pricing()
            TimeToMaturity_Pricing=input.TimeToMaturity_Pricing()
            Yield_Pricing=input.Yield_Pricing()
            Approach_Pricing=input.Approach_Pricing()
         
        NumberOfSimulations=input.NumberOfSimulations()
        NumberOfSeeds=input.NumberOfSeeds()
        ImportanceSampling=input.ImportanceSampling()
        Rand_seed=input.Rand_seed()    
        AntitheticSampling=input.AntitheticSampling()
        ScalingFactor=input.ScalingFactor()  
        AmplificationFactor=input.AmplificationFactor()
        EScontrib=input.EScontrib()
        EScontrib_sim=input.EScontrib_sim()
        EScontrib_analytic=input.EScontrib_analytic()
        CondPDLGD=input.CondPDLGD()
        MRcontribSim=input.MRcontribSim()
        WCE=input.WCE()
        WCE_sim=input.WCE_sim()
        WCE_analytic=input.WCE_analytic()
        LossDistribution=input.LossDistribution()
        LossDistribution2=input.LossDistribution2()
        GraphicalOutput=input.GraphicalOutput()
        MarketValueApproach=input.MarketValueApproach()
        MaxPoolSize=input.MaxPoolSize()
        LGD_Pool=input.LGD_Pool()
        LGD_Pool_min=input.LGD_Pool_min()
        if not LGD_Pool:
            LGD_Pool_min=1 # Force pooled LGD simulation
        ParallelComp=input.ParallelComp()
        NumberOfCores=input.NumberOfCores()
        BatchSize_Sim=input.BatchSize_Sim()
        BatchSize_Sim1=input.BatchSize_Sim1()
        
        Quantiles1=input.Quantiles1()
        Quantiles2=input.Quantiles2()
        Quantiles3=input.Quantiles3()
        Quantiles4=input.Quantiles4()
        Quantiles5=input.Quantiles5()
        Quantiles = [0] * input.NumberOfCIs()  
        Quantiles[0] = input.Quantiles1()  
        if input.NumberOfCIs()>=2:
            Quantiles[1] = input.Quantiles2()  
        if input.NumberOfCIs()>=3:
            Quantiles[2] = input.Quantiles3()   
        if input.NumberOfCIs()>=4:
            Quantiles[3] = input.Quantiles4()  
        if input.NumberOfCIs()>=5:
            Quantiles[4] = input.Quantiles5()
        Quantiles = [q / 100 for q in Quantiles] 

        # Preparation of variables for capital allocation
        if RunType2 or RunType3:

            ESType1=False
            ESType2=False
            ESLowerBoundary=False
            
            if input.ESType()=="1":
                ESType1=True
            elif input.ESType()=="2":
                ESType2=True
            elif input.ESType()=="3":
                ESLowerBoundary=True
                LowerBoundary_ES=input.LowerBoundary_ES()
            
            AllocationQuantile1=input.AllocationQuantile1()
            AllocationQuantile2=input.AllocationQuantile2()
            AllocationQuantile3=input.AllocationQuantile3()
            AllocationQuantile4=input.AllocationQuantile4()
            AllocationQuantile5=input.AllocationQuantile5()
            
            WCE_window1 = True
            WCE_window2 = False
            WCE_window3 = False
            WCE_window4 = False
            WCE_window5 = False

            # Initialize arrays
            UpperBoundary = [0] * 5
            LowerBoundary = [0] * 5

            # Set first window boundaries
            UpperBoundary[0] = input.UpperBoundary1()
            LowerBoundary[0] = input.LowerBoundary1()

            # Check how many windows are requested and assign accordingly
            if input.NumberOfWindows() >= 2:
                WCE_window2 = True
                UpperBoundary[1] = input.UpperBoundary2()
                LowerBoundary[1] = input.LowerBoundary2()

                if input.NumberOfWindows() >= 3:
                    WCE_window3 = True
                    UpperBoundary[2] = input.UpperBoundary3()
                    LowerBoundary[2] = input.LowerBoundary3()

                    if input.NumberOfWindows() >= 4:
                        WCE_window4 = True
                        UpperBoundary[3] = input.UpperBoundary4()
                        LowerBoundary[3] = input.LowerBoundary4()

                        if input.NumberOfWindows() >= 5:
                            WCE_window5 = True
                            UpperBoundary[4] = input.UpperBoundary5()
                            LowerBoundary[4] = input.LowerBoundary5()
            
            UpperBoundary = [q / 100 for q in UpperBoundary] 
            LowerBoundary = [q / 100 for q in LowerBoundary] 
                    
            NumberOfAllocationQuantiles = 1
            NumberOfWCE_windows = 1
        
            if len(Quantiles) == 1:
                NumberOfAllocationQuantiles = max(1, AllocationQuantile1)
                NumberOfWCE_windows = max(1, WCE_window1)
            elif len(Quantiles) == 2:
                NumberOfAllocationQuantiles = max(1, AllocationQuantile1 + AllocationQuantile2)
                NumberOfWCE_windows = max(1, WCE_window1 + WCE_window2)
            elif len(Quantiles) == 3:
                NumberOfAllocationQuantiles = max(1, AllocationQuantile1 + AllocationQuantile2 + AllocationQuantile3)
                NumberOfWCE_windows = max(1, WCE_window1 + WCE_window2 + WCE_window3)
            elif len(Quantiles) == 4:
                NumberOfAllocationQuantiles = max(1, AllocationQuantile1 + AllocationQuantile2 + AllocationQuantile3 + AllocationQuantile4)
                NumberOfWCE_windows = max(1, WCE_window1 + WCE_window2 + WCE_window3 + WCE_window4)
            elif len(Quantiles) == 5:
                NumberOfAllocationQuantiles = max(1, AllocationQuantile1 + AllocationQuantile2 + AllocationQuantile3 + AllocationQuantile4 + AllocationQuantile5)
                NumberOfWCE_windows = max(1, WCE_window1 + WCE_window2 + WCE_window3 + WCE_window4 + WCE_window5)

             # Safety fallback
            if AllocationQuantile1 + AllocationQuantile2 + AllocationQuantile3 + AllocationQuantile4 + AllocationQuantile5 == 0:
                AllocationQuantile1 = True 

            if WCE_window1 + WCE_window2 + WCE_window3 + WCE_window4 + WCE_window5 == 0:
                WCE_window1 = 1

            AllocationQuantiles = [AllocationQuantile1, AllocationQuantile2, AllocationQuantile3, AllocationQuantile4, AllocationQuantile5]
            AllocationQuantiles = AllocationQuantiles[:len(Quantiles)]
          
            # Sort Quantiles by AllocationQuantiles in decreasing order
            Quantiles = [q for _, q in sorted(zip(AllocationQuantiles, Quantiles), reverse=True)]

            # Swap boundaries if lower > upper
            for j in range(5):
                if LowerBoundary[j] > UpperBoundary[j]:
                    LowerBoundary[j], UpperBoundary[j] = UpperBoundary[j], LowerBoundary[j]

            # Sort boundaries based on active WCE_windows in descending order
            WCE_windows = [WCE_window1, WCE_window2, WCE_window3, WCE_window4, WCE_window5]
            # Convert booleans to integers (True = 1, False = 0) for sorting
            sort_order = sorted(range(5), key=lambda i: int(WCE_windows[i]), reverse=True)

            UpperBoundary = [UpperBoundary[i] for i in sort_order]
            LowerBoundary = [LowerBoundary[i] for i in sort_order]



         

        ###############################
        ### Handling of file inputs ###
        ###############################

        ui.notification_show(
            f"Start data loading and data processing at {datetime.now().strftime('%H:%M:%S')} ...",
            type="message",  # Could also be "default", "warning", "error"
            duration=5
        )   
        file1=input.file1()
        PortfolioData = pd.read_csv(file1[0]["datapath"],sep=";")
        #PortfolioData = pd.read_csv('Portfolio_PLC_Pool.csv',sep=";")

        #Checking of portfolio data
        NumberOfWarnings=0

        required_columns = {
            "ExposureID", "GroupID", "EAD", "LGD", "PD", "Number_of_Exposures", "Segment",
            "Rsquared", "LGD_Segment", "k_Parameter", "LGD_Rsquared", "Approach",
            "RatingClass", "TimeToMaturity", "EIR", 
        }
        missing_columns = required_columns - set(PortfolioData.columns)
        if len(missing_columns) > 0:
            ui.notification_show(f"Portfolio data does not provide the following information {missing_columns}. Please use the standard template.", type="error",duration=None)
            NumberOfWarnings+=1
        
        
        
        try:
            if np.min(PortfolioData['EAD'])<=0:
                ui.notification_show("Portfolio data contains entries with non-positive EAD.", type="error",duration=None)
                NumberOfWarnings+=1
            if np.min(PortfolioData['LGD'])<=0:
                ui.notification_show("Portfolio data contains entries with non-positive LGD.", type="error",duration=None)
                NumberOfWarnings+=1
            if np.max(PortfolioData['LGD'])>1:
                ui.notification_show("Portfolio data contains LGD parameters greater than 100%.", type="error",duration=None)
                NumberOfWarnings+=1
            if np.min(PortfolioData['PD'])<=0:
                ui.notification_show("Portfolio data contains entries with non-positive PD.", type="error",duration=None)
                NumberOfWarnings+=1
            if np.max(PortfolioData['PD'])>1:
                ui.notification_show("Portfolio data contains PD parameters greater than 100%.", type="error",duration=None)
                NumberOfWarnings+=1
            if np.min(PortfolioData['Rsquared'])<0:
                ui.notification_show("Portfolio data contains entries with negative R-squared parameters.", type="error",duration=None)
                NumberOfWarnings+=1
            if np.max(PortfolioData['Rsquared'])>=1:
                ui.notification_show("Portfolio data contains R-squared parameters of more than 100%.", type="error",duration=None)
                NumberOfWarnings+=1
            if Vasicek or PLC:
                if np.min(PortfolioData['LGD_Rsquared'])<0:
                    ui.notification_show("Portfolio data contains entries with negative LGD R-squared parameters.", type="error",duration=None)
                    NumberOfWarnings+=1
                if np.max(PortfolioData['LGD_Rsquared'])>=1:
                    ui.notification_show("Portfolio data contains LGD R-squared parameters of more than 100%.", type="error",duration=None)
                    NumberOfWarnings+=1
                if np.min(PortfolioData['k_Parameter'])<=1:
                    ui.notification_show("Portfolio data contains k-parameters that don't exceed 1.", type="error",duration=None)
                    NumberOfWarnings+=1
            if np.min(PortfolioData['Number_of_Exposures'])<=0:
                ui.notification_show("Portfolio data contains entries with non-positive number of exposures.", type="error",duration=None)
                NumberOfWarnings+=1
            if not np.all(np.isclose(PortfolioData['Number_of_Exposures'] % 1, 0)):
                ui.notification_show("Portfolio data contains entries with non-integer number of exposures.", type="error",duration=None)
                NumberOfWarnings+=1
            
            # Identify duplicate GroupIDs
            duplicate_ids = PortfolioData['GroupID'][PortfolioData['GroupID'].duplicated(keep=False)]
            # Filter rows with non-unique GroupIDs
            non_unique_groups = PortfolioData[PortfolioData['GroupID'].isin(duplicate_ids)]
            if np.max(non_unique_groups['Number_of_Exposures'])>1:
                ui.notification_show("Some of the Group IDs for exposure pools have multiple entries although these should be unique.", type="error",duration=None)
                NumberOfWarnings+=1
        except(ValueError, TypeError):
            ui.notification_show("Portfolio data contains non-numeric information for relevant parameters.", type="error",duration=None)
            NumberOfWarnings+=1
        
        if NumberOfWarnings>0:
            ui.notification_show("Please correct these issues before rerunning the model.", type="error",duration=None)    
            return  # Stop further execution in case of critical issues



        # Grouping of exposures

        PortfolioData['GroupedExposure'] = "No"
        # Replace empty GroupID with corresponding ExposureID
        empty_group_ids = PortfolioData['GroupID'] == ""
        PortfolioData.loc[empty_group_ids, 'GroupID'] = PortfolioData.loc[empty_group_ids, 'ExposureID']

        group_counts = PortfolioData['GroupID'].value_counts()
        grouped_ids = group_counts[group_counts > 1].index
        PortfolioData.loc[PortfolioData['GroupID'].isin(grouped_ids), 'GroupedExposure'] = "Yes"

        GroupIDs = PortfolioData['GroupID'].unique()

                
        # Convert correlation matrix into factor table in order to simulate correlated random numbers
        if input.Cor_Input_Type()=="1":
            file2=input.file2()
            CorrelationMatrix_df = pd.read_csv(file2[0]["datapath"],sep=";",header=0,index_col=0)
            #CorrelationMatrix_df = pd.read_csv('CorrelationMatrix_New.csv',sep=";",header=0,index_col=0)
            SectorNames=list(CorrelationMatrix_df.columns)
            CorrelationMatrix = CorrelationMatrix_df.to_numpy()

            # Quality checks for correlation matrix
            portfolio_segments = PortfolioData["Segment"].unique()
            combined = pd.Series(list(portfolio_segments) + list(CorrelationMatrix_df.index)) #Combined set of segments
            portfolio_segments_set = set(portfolio_segments)
            matrix_segments = set(CorrelationMatrix_df.index)

            missing_segments = portfolio_segments_set - matrix_segments
                        
            NumberOfWarnings=0
            if list(CorrelationMatrix_df.index)!=list(CorrelationMatrix_df.columns):
                ui.notification_show("Correlation matrix has different segments in rows vs. columns.", type="error",duration=None)
                NumberOfWarnings+=1
            if combined.nunique() > pd.Series(CorrelationMatrix_df.index).nunique():
                ui.notification_show(f"Correlation matrix does not cater for segments {missing_segments}.", type="error",duration=None)
                NumberOfWarnings+=1
    
            try:
                if np.max(CorrelationMatrix)>1:
                    ui.notification_show("Correlation matrix contains entries >100%.", type="error",duration=None)
                    NumberOfWarnings+=1
                if np.min(CorrelationMatrix)<-1:
                    ui.notification_show("Correlation matrix contains entries <-100%.", type="error",duration=None)
                    NumberOfWarnings+=1
                if not np.allclose(CorrelationMatrix, CorrelationMatrix.T):
                    ui.notification_show("Correlation matrix is not symmetric.", type="error",duration=None)
                    NumberOfWarnings+=1
                else:
                    if not np.all(np.linalg.eigvalsh(CorrelationMatrix) >= -1e-8):
                        ui.notification_show("Correlation matrix is not positive semidefinite. Negative eigenvalues are replaces with 0.", type="warning",duration=5) 
                if not np.allclose(np.diag(CorrelationMatrix), 1.0):
                    ui.notification_show("Some diagonal elements of the correlation matrix differ from 100%.", type="error",duration=None)
                    NumberOfWarnings+=1
            except(ValueError, TypeError):
                ui.notification_show("Correlation matrix contains non-numeric values.", type="error",duration=None)
                NumberOfWarnings+=1

            if PLC:
                portfolio_lgd_segments = PortfolioData["LGD_Segment"].unique()
                combined = pd.Series(list(portfolio_lgd_segments) + list(CorrelationMatrix_df.index)) #Combined set of segments
                portfolio_lgd_segments_set = set(portfolio_lgd_segments)
                
                missing_lgd_segments = portfolio_lgd_segments_set - matrix_segments
                                
                if combined.nunique() > pd.Series(CorrelationMatrix_df.index).nunique():
                    ui.notification_show(f"Correlation matrix does not cater for LGD segments {missing_lgd_segments}.", type="error",duration=None)
                    NumberOfWarnings+=1

            if NumberOfWarnings>0:
                ui.notification_show("Please correct these issues before rerunning the model.", type="error",duration=None)    
                return  # Stop further execution in case of critical issues 
            
            
            # Perform eigen decomposition and output eigenvalues in descending order
            eigenvalues, eigenvectors = np.linalg.eigh(CorrelationMatrix)  # Since it's symmetric
            eigenvalues=eigenvalues[::-1] 
            eigenvectors=eigenvectors[:, ::-1]

            # Correct small negative eigenvalues (if any)
            eigen_adj = np.maximum(eigenvalues, 0)

            # Create diagonal matrix of sqrt(eigen_adj)
            C = np.sqrt(np.diag(eigen_adj))

            # Compute Factor Table
            FactorTable = eigenvectors @ C

            # Round to 8 decimals
            FactorTable = np.round(FactorTable, 8)
            # To ensure the same factor table for each run
            for i in range(FactorTable.shape[1]):
                col = FactorTable[:, i]
                if -np.min(col) > np.max(col):
                    FactorTable[:, i] = -col
                elif -np.min(col) == np.max(col):
                    if np.argmax(col) > np.argmin(col):
                        FactorTable[:, i] = -col

        else:
            file2_factor=input.file2_factor()
            FactorTable_df = pd.read_csv(file2_factor[0]["datapath"],sep=";",header=0,index_col=0)
            SectorNames=list(FactorTable_df.index)
            FactorTable = FactorTable_df.to_numpy()

            # Quality checks for factor table
            portfolio_segments = PortfolioData["Segment"].unique()
            combined = pd.Series(list(portfolio_segments) + list(FactorTable_df.index)) #Combined set of segments
            portfolio_segments_set = set(portfolio_segments)
            matrix_segments = set(FactorTable_df.index)

            missing_segments = portfolio_segments_set - matrix_segments
                        
            NumberOfWarnings=0
            if combined.nunique() > pd.Series(FactorTable_df.index).nunique():
                ui.notification_show(f"Factor table does not cater for segments {missing_segments}.", type="error",duration=None)
                NumberOfWarnings+=1
    
            try:
                if np.allclose(np.sum(FactorTable**2, axis=1), 1.0): 
                    ui.notification_show("Weights for some segments are not standardised. Linear scaling will be applied.", type="warning",duration=5)
            except(ValueError, TypeError):
                ui.notification_show("Factor table contains non-numeric values.", type="error",duration=None)
                NumberOfWarnings+=1

            if PLC:
                portfolio_lgd_segments = PortfolioData["LGD_Segment"].unique()
                combined = pd.Series(list(portfolio_lgd_segments) + list(FactorTable_df.index)) #Combined set of segments
                portfolio_lgd_segments_set = set(portfolio_lgd_segments)
                
                missing_lgd_segments = portfolio_lgd_segments_set - matrix_segments
                                
                if combined.nunique() > pd.Series(FactorTable_df.index).nunique():
                    ui.notification_show(f"Factor table does not cater for LGD segments {missing_lgd_segments}.", type="error",duration=None)
                    NumberOfWarnings+=1

            if NumberOfWarnings>0:
                ui.notification_show("Please correct these issues before rerunning the model.", type="error",duration=None)    
                return  # Stop further execution in case of critical issues 
            
            for i in range(FactorTable.shape[0]):
                FactorTable[i, :] = FactorTable[i, :] / np.sum(FactorTable[i, :] ** 2)

        FactorTable = pd.DataFrame(FactorTable, columns=list(range(FactorTable.shape[1])), index=SectorNames)
        NumberOfFactors = FactorTable.shape[1]
        
        # Convert LGD correlation matrix into factor table in order to simulate correlated random numbers
        if Vasicek==True:
            file3=input.file3()
            LGDCorrelationMatrix_df = pd.read_csv(file3[0]["datapath"],sep=";",header=0,index_col=0)
            LGDSectorNames=list(LGDCorrelationMatrix_df.columns)
            LGDCorrelationMatrix = LGDCorrelationMatrix_df.to_numpy()

            # Quality checks for LGD correlation matrix
            portfolio_lgd_segments = PortfolioData["LGD_Segment"].unique()
            combined = pd.Series(list(portfolio_lgd_segments) + list(LGDCorrelationMatrix_df.index)) #Combined set of segments
            portfolio_lgd_segments_set = set(portfolio_lgd_segments)
            matrix_lgd_segments = set(LGDCorrelationMatrix_df.index)

            missing_lgd_segments = portfolio_lgd_segments_set - matrix_lgd_segments
                        
            NumberOfWarnings=0
            if list(LGDCorrelationMatrix_df.index)!=list(LGDCorrelationMatrix_df.columns):
                ui.notification_show("LGD correlation matrix has different segments in rows vs. columns.", type="error",duration=None)
                NumberOfWarnings+=1
            if combined.nunique() > pd.Series(LGDCorrelationMatrix_df.index).nunique():
                ui.notification_show(f"LGD correlation matrix does not cater for segments {missing_lgd_segments}.", type="error",duration=None)
                NumberOfWarnings+=1
    
            try:
                if np.max(LGDCorrelationMatrix)>1:
                    ui.notification_show("LGD correlation matrix contains entries >100%.", type="error",duration=None)
                    NumberOfWarnings+=1
                if np.min(LGDCorrelationMatrix)<-1:
                    ui.notification_show("LGD correlation matrix contains entries <-100%.", type="error",duration=None)
                    NumberOfWarnings+=1
                if not np.allclose(LGDCorrelationMatrix, LGDCorrelationMatrix.T):
                    ui.notification_show("LGD correlation matrix is not symmetric.", type="error",duration=None)
                    NumberOfWarnings+=1
                else:
                    if not np.all(np.linalg.eigvalsh(LGDCorrelationMatrix) >= -1e-8):
                        ui.notification_show("LGD correlation matrix is not positive semidefinite. Negative eigenvalues are replaces with 0.", type="warning",duration=5) 
                if not np.allclose(np.diag(LGDCorrelationMatrix), 1.0):
                    ui.notification_show("Some diagonal elements of the LGD correlation matrix differ from 100%.", type="error",duration=None)
                    NumberOfWarnings+=1
            except(ValueError, TypeError):
                ui.notification_show("LGD correlation matrix contains non-numeric values.", type="error",duration=None)
                NumberOfWarnings+=1
          
            if NumberOfWarnings>0:
                ui.notification_show("Please correct these issues before rerunning the model.", type="error",duration=None)    
                return  # Stop further execution in case of critical issues 

            
            # Perform eigen decomposition and output eigenvalues in descending order
            eigenvalues, eigenvectors = np.linalg.eigh(LGDCorrelationMatrix)  # Since it's symmetric
            eigenvalues=eigenvalues[::-1] 
            eigenvectors=eigenvectors[:, ::-1]

            # Correct small negative eigenvalues (if any)
            eigen_adj = np.maximum(eigenvalues, 0)

            # Create diagonal matrix of sqrt(eigen_adj)
            C = np.sqrt(np.diag(eigen_adj))

            # Compute Factor Table
            LGDFactorTable = eigenvectors @ C

            # Round to 8 decimals
            LGDFactorTable = np.round(LGDFactorTable, 8)
            # To ensure the same factor table for each run
            for i in range(LGDFactorTable.shape[1]):
                col = LGDFactorTable[:, i]
                if -np.min(col) > np.max(col):
                    LGDFactorTable[:, i] = -col
                elif -np.min(col) == np.max(col):
                    if np.argmax(col) > np.argmin(col):
                        LGDFactorTable[:, i] = -col

            LGDFactorTable = pd.DataFrame(LGDFactorTable, columns=list(range(LGDFactorTable.shape[1])), index=LGDSectorNames)
            NumberOfLGDFactors = LGDFactorTable.shape[1]
        
        # To ensure even number of simulations for antithetic sampling
        if AntitheticSampling:
            NumberOfSimulations = 2 * round(NumberOfSimulations / 2)

        NumberOfPools=PortfolioData.shape[0]
        
        # Determination of optimal mean shift according to Kalkbrener
        if not RunType3:
            if ImportanceSampling and ISType2:
                help_vector = np.zeros((NumberOfFactors, 1))
                for i in range(0,NumberOfPools):
                    row = PortfolioData.iloc[i]
                    segment = row['Segment']
                    help_vector += (
                        FactorTable.loc[segment, :].to_numpy().reshape(-1, 1)
                        * row['EAD']
                        * row['LGD']
                        * row['PD']
                        * np.sqrt(row['Rsquared'])
                    )

                EAD = PortfolioData['EAD'].to_numpy()
                LGD = PortfolioData['LGD'].to_numpy()
                PD = PortfolioData['PD'].to_numpy()
                Rsquared = PortfolioData['Rsquared'].to_numpy()
                NoE = PortfolioData['Number_of_Exposures'].to_numpy()

                A = np.sum(help_vector**2)
                B = np.sum((1 - Rsquared) / NoE * (EAD * LGD * PD) ** 2)
                C = np.sum((EAD / NoE * LGD * PD * np.sqrt(Rsquared))   ** 2 * NoE)
                D = np.sum(EAD * LGD * PD) ** 2
                E = np.sum((EAD / NoE * LGD * PD) ** 2 * NoE)

                RSQ_hom = (A + B - C) / (D - E)
                PD_hom = np.sum(EAD * LGD * PD) / np.sum(EAD * LGD)

                q_star = quantile_search(Quantiles[0], PD_hom, RSQ_hom, 1e-10, 10000)

                result = minimize_scalar(
                    lambda mu: optimal_shift(mu, q_star, PD_hom, RSQ_hom, 1_000_000),
                    bounds=(-20, 20),
                    method='bounded'
                )

                mu_optimal = result.x

                s_optimal = np.sqrt(
                    1 / RSQ_hom * (np.sum(help_vector**2) + np.sum((1 - Rsquared) / NoE * (EAD * LGD * PD) ** 2))
                )

                rho_optimal = help_vector * (1 / s_optimal)

                Mean_shift = np.zeros((NumberOfFactors, 1))
                for k in range(NumberOfFactors):
                    Mean_shift[k] = mu_optimal * rho_optimal[k] / np.sqrt(RSQ_hom)

                Mean_shift *= AmplificationFactor

                if ISType3 and NumberOfFactors > 1:
                    Mean_shift[1:] = 0

        # Preparation of stochastic LGD modelling
        if not Deterministic:
            PortfolioData["rho"] = 0.0
            # Step 1: Create a mask for the rows you want to modify
            mask = (PortfolioData['LGD'] > 0) & (PortfolioData['LGD'] < 1)

            # Step 2: Compute the indices for the RhoMapping lookup
            lgd = PortfolioData.loc[mask, 'LGD']
            k_param = PortfolioData.loc[mask, 'k_Parameter']

            row_idx = np.ceil(2000 * np.sqrt(lgd * (1 - lgd) / k_param) - 0.5).astype(int)
            col_idx = np.ceil(2000 * lgd - 0.5).astype(int)

            # Step 3: Look up values in RhoMapping
            rho_values = RhoMapping[row_idx, col_idx]

            # Step 4: Clip rho values to maximum 0.9999
            rho_values = np.minimum(rho_values, 0.9999)

            # Step 5: Assign back
            PortfolioData.loc[mask, 'rho'] = rho_values

            # Alternatively, this could be recalculated: AC_fit(lgd,lgd, lgd*(1-lgd)/PortfolioData.k_Parameter.iloc[i]+lgd**2, tolerance=1e-12, max_iter=100)

        # Calculation of conditional expectation and variance for systematic recovery factors
        if PLC:
            def compute_mu_PLC(row):
                rsq = row['Rsquared']
                pd = row['PD']
                
                if pd >= 1:
                    return 0.0
                
                seg1 = row['Segment']
                seg2 = row['LGD_Segment']
                
                vec1 = FactorTable.loc[seg1]
                vec2 = FactorTable.loc[seg2]

                dot_product = np.sum(vec1 * vec2)

                qn = ndtri(pd)
                density = np.exp(-0.5 * qn**2) / (pd * np.sqrt(2 * np.pi))

                rho = -np.sqrt(rsq) * dot_product * density
                return rho
            
            PortfolioData['mu_PLC'] = PortfolioData.apply(compute_mu_PLC, axis=1)

            def compute_sigma_PLC(row):
                pd = row['PD']
                rsq = row['Rsquared']

                if pd >= 1:
                    return 1.0

                seg1 = row['Segment']
                seg2 = row['LGD_Segment']

                vec1 = FactorTable.loc[seg1]
                vec2 = FactorTable.loc[seg2]

                dot_product = np.sum(vec1 * vec2)
                dot_squared = dot_product ** 2
                qn = ndtri(pd)

                # Components
                term1 = (qn * rsq * dot_squared) / (pd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * qn**2)
                term2 = (rsq * dot_squared) / ((pd**2) * 2 * np.pi) * np.exp(-qn**2)

                result = 1 - term1 - term2
                return result

            PortfolioData["sigma_PLC"] = PortfolioData.apply(compute_sigma_PLC, axis=1)
            
        if not MarketValueApproach:
            PortfolioData["Approach"] ="Default"
        
        if MarketValueApproach:
            file4=input.file4()
            MigrationMatrix_df = pd.read_csv(file4[0]["datapath"],sep=";",header=0,index_col=0)
            #MigrationMatrix_df = pd.read_csv('MigrationMatrix_New.csv',sep=";",header=0,index_col=0)
            RatingClasses=list(MigrationMatrix_df.columns)
            MigrationMatrix = MigrationMatrix_df.to_numpy()
            
            # Quality checks for migration matrix
            market_ratings = PortfolioData.loc[PortfolioData["Approach"] == "Market value", "RatingClass"]
            combined = pd.Series(list(market_ratings) + list(MigrationMatrix_df.index)) #Combined set of rating classes
            portfolio_ratings = set(market_ratings.dropna().astype(str).unique())
            matrix_ratings = set(MigrationMatrix_df.index.astype(str))

            missing_ratings = portfolio_ratings - matrix_ratings
            NumberOfWarnings=0
         
            try:
                if list(MigrationMatrix_df.index.astype(int))!=list(range(1,MigrationMatrix_df.shape[0]+1)) or list(MigrationMatrix_df.columns.astype(int))!=list(range(1,MigrationMatrix_df.shape[0]+1)):
                    ui.notification_show("Rating classes in migration matrix are not aligned to standard numbering 1, 2, ..., N.", type="error",duration=None)
                    NumberOfWarnings+=1
                if combined.nunique() > pd.Series(MigrationMatrix_df.index).nunique():
                    ui.notification_show(f"Migration matrix does not cater for rating classes {missing_ratings}.", type="error",duration=None)
                    NumberOfWarnings+=1
                if list(MigrationMatrix_df.index.astype(str))!=list(MigrationMatrix_df.columns.astype(str)):
                    ui.notification_show("Migration matrix has different rating classes in rows vs. columns.", type="error",duration=None)
                    NumberOfWarnings+=1
            except(ValueError, TypeError):
                ui.notification_show("Rating classes in migration matrix contain non-integer values.", type="error",duration=None)
                NumberOfWarnings+=1
            try:
                if np.max(MigrationMatrix)>1:
                    ui.notification_show("Migration matrix contains entries >100%.", type="error",duration=None)
                    NumberOfWarnings+=1
                if np.min(MigrationMatrix)<0:
                    ui.notification_show("Migration matrix contains negative entries.", type="error",duration=None)
                    NumberOfWarnings+=1
                if np.min(np.sum(MigrationMatrix, axis=1))<(1-(1e-10)) or np.max(np.sum(MigrationMatrix, axis=1))>(1+(1e-10)):
                    ui.notification_show("Migration matrix does not fulfil row sum = 1 requirement. Pro rata scaling is applied to correct this.", type="warning",duration=5)
            except(ValueError, TypeError):
                ui.notification_show("Migration matrix contains non-numeric values.", type="error",duration=None)
                NumberOfWarnings+=1
                
            if NumberOfWarnings>0:
                ui.notification_show("Please correct these issues before rerunning the model.", type="error",duration=None)    
                return  # Stop further execution in case of critical issues  
   





            
            #Determination of migration boundaries
            MigrationMatrix = MigrationMatrix / MigrationMatrix.sum(axis=1, keepdims=True) #To ensure row sums of 1
            Migration_boundaries = MigrationMatrix.copy()
            for i in range(1,MigrationMatrix.shape[0]+1):
                cumulative = np.clip(MigrationMatrix[:, -i:].sum(axis=1),0,1)
                Migration_boundaries[:,-i]=ndtri(cumulative)

            MaxDuration=int(np.max(np.ceil(PortfolioData["TimeToMaturity"])))
            MigrationMatrix_cum=MigrationMatrix.copy()

            #Derivation of cumulative PD
            CumulativePD=np.zeros((MigrationMatrix.shape[0], MaxDuration))
            CumulativePD[:,0]=MigrationMatrix[:,-1]
            MarginalPD=np.zeros((MigrationMatrix.shape[0], MaxDuration))
            MarginalPD[:,0]=MigrationMatrix[:,-1]

            if MaxDuration>=2:
                for t in range(MaxDuration-1):
                    MigrationMatrix_cum=MigrationMatrix_cum @ MigrationMatrix
                    CumulativePD[:,t+1]=MigrationMatrix_cum[:,-1]
                    MarginalPD[:,t+1]=CumulativePD[:,t+1]-CumulativePD[:,t]

            for i in range(1, MigrationMatrix.shape[0]+1):
                PortfolioData[f"Migration effect rating class {i}"] = 0.0 # Columns for migration effects

            for i in range(PortfolioData.shape[0]):
                if PortfolioData.Approach.iloc[i]=="Market value":
                    PortfolioData.loc[PortfolioData.index[i], 'PD']=MigrationMatrix[int(PortfolioData.RatingClass.iloc[i])-1,-1] #Makes PD consistent to migration matrix
                    if PortfolioData.TimeToMaturity.iloc[i]<=1:
                        LifetimeLoss=PortfolioData.EAD.iloc[i]*PortfolioData.PD.iloc[i]*PortfolioData.LGD.iloc[i]
                        for j in range(1, MigrationMatrix.shape[0]):
                            PortfolioData.iloc[i, PortfolioData.columns.get_loc(f"Migration effect rating class {j}")] = PortfolioData.EAD.iloc[i]*PortfolioData.LGD.iloc[i]*CumulativePD[j-1,0] - LifetimeLoss
                        PortfolioData.iloc[i, PortfolioData.columns.get_loc(f"Migration effect rating class {MigrationMatrix.shape[0]}")] = PortfolioData.EAD.iloc[i]*PortfolioData.LGD.iloc[i] # Default event
                    else:
                        eir = PortfolioData.EIR.iloc[i]
                        ttm = int(np.floor(PortfolioData.TimeToMaturity.iloc[i]))
                        rating = int(PortfolioData.RatingClass.iloc[i])
                        timesteps = np.arange(ttm)

                        LifetimeLoss = np.sum(np.exp(-eir * timesteps) * MarginalPD[rating-1, timesteps]) 
                        if PortfolioData.TimeToMaturity.iloc[i]>np.floor(PortfolioData.TimeToMaturity.iloc[i]):
                            LifetimeLoss += np.exp(-eir*ttm)*MarginalPD[rating-1, ttm]*(PortfolioData.TimeToMaturity.iloc[i]-np.floor(PortfolioData.TimeToMaturity.iloc[i]))
                        LifetimeLoss *= PortfolioData.EAD.iloc[i]*PortfolioData.LGD.iloc[i]
                        
                        for j in range(1, MigrationMatrix.shape[0]):
                            LifetimeLoss_Rating=np.sum(np.exp(-eir * timesteps) * MarginalPD[j-1, timesteps])
                            if PortfolioData.TimeToMaturity.iloc[i]>np.floor(PortfolioData.TimeToMaturity.iloc[i]):
                                LifetimeLoss_Rating += np.exp(-eir*ttm)*MarginalPD[j-1, ttm]*(PortfolioData.TimeToMaturity.iloc[i]-np.floor(PortfolioData.TimeToMaturity.iloc[i]))
                            LifetimeLoss_Rating *= PortfolioData.EAD.iloc[i]*PortfolioData.LGD.iloc[i]
                            PortfolioData.iloc[i, PortfolioData.columns.get_loc(f"Migration effect rating class {j}")] = LifetimeLoss_Rating - LifetimeLoss
                        PortfolioData.iloc[i, PortfolioData.columns.get_loc(f"Migration effect rating class {MigrationMatrix.shape[0]}")] = PortfolioData.EAD.iloc[i]*PortfolioData.LGD.iloc[i] # Default event

            col_names = [f"Migration effect rating class {i+1}" for i in range(MigrationMatrix.shape[0])]
            MigrationEffects=PortfolioData[col_names].to_numpy() #Array to be used in simulations
            MigrationEffects[:,-1]=0.0 #Losses in default events are subject to separate treatment
            MigrationEffects[-1,:]=0.0 #Clients already in default are not subject ot migration effects

        #PortfolioData.to_csv("PortfolioData.csv", index=True)
       
        ########################################
        ### Initialisation of summary tables ###
        ########################################


        PortfolioLoss = np.zeros((NumberOfSimulations, NumberOfSeeds * (1 + ImportanceSampling)))
        if MarketValueApproach: #Separation of losses due to default and migration events
             PortfolioLoss_migration = np.zeros((NumberOfSimulations, NumberOfSeeds))
             PortfolioLoss_default = np.zeros((NumberOfSimulations, NumberOfSeeds))

        
        EL_simulated = np.zeros((NumberOfSeeds, 1))
        StdDev_simulated = np.zeros((NumberOfSeeds, 1))
        VaR = pd.DataFrame(np.zeros((NumberOfSeeds, len(Quantiles))), columns=Quantiles)
        ES = np.zeros((NumberOfSeeds, len(Quantiles)))
        ECAP = np.zeros((NumberOfSeeds, len(Quantiles)))
        if MarketValueApproach:
            EL_simulated_default = np.zeros((NumberOfSeeds, 1))
            StdDev_simulated_default = np.zeros((NumberOfSeeds, 1))
            VaR_default = pd.DataFrame(np.zeros((NumberOfSeeds, len(Quantiles))), columns=Quantiles)
            ES_default = np.zeros((NumberOfSeeds, len(Quantiles)))
            ECAP_default = np.zeros((NumberOfSeeds, len(Quantiles)))

            EL_simulated_migration = np.zeros((NumberOfSeeds, 1))
            StdDev_simulated_migration = np.zeros((NumberOfSeeds, 1))
            VaR_migration = pd.DataFrame(np.zeros((NumberOfSeeds, len(Quantiles))), columns=Quantiles)
            ES_migration = np.zeros((NumberOfSeeds, len(Quantiles)))
            ECAP_migration = np.zeros((NumberOfSeeds, len(Quantiles)))

              
        # Analytical Expected Loss calculation
        EL_analytical = PortfolioData['EAD'] * PortfolioData['PD'] * PortfolioData['LGD']
        if MarketValueApproach:
            mv_mask = PortfolioData['Approach'] == "Market value"
            rating_indices = PortfolioData.loc[mv_mask, 'RatingClass'].astype(int) - 1
            EL_analytical_default=EL_analytical.copy()
            EL_analytical_migration=np.zeros_like(EL_analytical_default)

            for idx, rating in zip(PortfolioData.index[mv_mask], rating_indices):
                EL_analytical_migration[idx] = np.dot(MigrationEffects[idx, :], MigrationMatrix[rating, :])
            EL_analytical += EL_analytical_migration
        
        RunTime = datetime.now() - StartDate  # This is a timedelta object

        hours, remainder = divmod(RunTime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        ui.notification_show(
            f"Data loading and data processing finished after {int(hours)}h:{int(minutes)}m:{int(round(seconds))}s",
            type="message",
            duration=5,
        )


        ########################################
        ### Capital window runs              ###
        ########################################

        StartTime_CW=datetime.now()  
        ui.notification_show(
            f"Start capital window runs at {StartTime_CW.strftime('%H:%M:%S')} ...",
            type="message",  # Could also be "default", "warning", "error"
            duration=5
        )        
                
        for run in range(1, NumberOfSeeds + 1):
            rng = np.random.default_rng(seed=Rand_seed+run)
            Progress_bar_counter=0
            with ui.Progress(min=1, max=PortfolioData.shape[0]) as Progress_bar:
                Progress_bar.set(message=f"Capital window run {run} in progress") 

                #Generate correlated systematic factors
                if not AntitheticSampling:
                    X = rng.normal(size=(NumberOfSimulations, NumberOfFactors))
                else:
                    X_half = rng.normal(size=(int(NumberOfSimulations / 2), NumberOfFactors))
                    X = np.vstack((X_half, -X_half))
                    del X_half
                
                if ImportanceSampling:
                    if ISType1:
                        X[:, 0] *= ScalingFactor  # Application of importance sampling on first factor
                        PortfolioLoss[:, NumberOfSeeds + run-1] = ScalingFactor * np.exp(-(1 - 1 / ScalingFactor**2) / 2 * (X[:, 0])**2)  # Determination of likelihood ratios
                    if ISType2:
                        for k in range(NumberOfFactors):
                            X[:, k] += Mean_shift[k]  # Application of mean shift
                        PortfolioLoss[:, NumberOfSeeds + run-1] = np.exp(Mean_shift[0] / 2 * (Mean_shift[0] - 2 * X[:, 0]))
                        if NumberOfFactors > 1:
                            for k in range(1, NumberOfFactors):
                                PortfolioLoss[:, NumberOfSeeds + run-1] *= np.exp(Mean_shift[k] / 2 * (Mean_shift[k] - 2 * X[:, k]))  # Determination of likelihood ratios
                    PortfolioLoss[:, NumberOfSeeds + run-1] /= NumberOfSimulations

            
                X=X @ FactorTable.T # Derivation of correlated systematic factor returns

                #Generate correlated systematic LGD factors (if PLC is not enabled)
                if Vasicek:
                    if not AntitheticSampling:
                        Y = rng.normal(size=(NumberOfSimulations, NumberOfLGDFactors))
                    else:
                        Y_half = rng.normal(size=(int(NumberOfSimulations / 2), NumberOfLGDFactors))
                        Y = np.vstack((Y_half, -Y_half))
                        del Y_half

                    Y=Y @ LGDFactorTable.T # Derivation of correlated systematic LGD factor returns

                if RunType3:
                    rng = np.random.default_rng(seed=Rand_seed+run+27644437) # To control random seed for pricing run

                start_time_progress = time.time()
            
                for group_id in GroupIDs:
                    # Filter rows where GroupID equals current group_id
                    group_data = PortfolioData[PortfolioData['GroupID'] == group_id]
                    
                    # Get max of Number_of_Exposures in that group
                    max_exposures = group_data['Number_of_Exposures'].max()
        
                    if max_exposures > 1: # Modelling for exposure pools
                        M=group_data.shape[0]
                        for i in range(0,M):
                            if Progress_bar_counter>0:
                                elapsed = time.time() - start_time_progress
                                est_total_time = elapsed / Progress_bar_counter * PortfolioData.shape[0]
                                eta_seconds = est_total_time - elapsed
                                eta_str = format_eta(eta_seconds)
                                Progress_bar.set(value=Progress_bar_counter,message=f"Capital window run {run}", detail=f"ETA {eta_str}") 
                            Progress_bar_counter+=1
                            if group_data.PD.iloc[i]<1:
                                if group_data.Number_of_Exposures.iloc[i] > MaxPoolSize:
                                    ConditionalPD = fast_norm_cdf(
                                        (ndtri(group_data.PD.iloc[i]) - np.sqrt(group_data.Rsquared.iloc[i]) * X[group_data.Segment.iloc[i]].values) / np.sqrt(1 - group_data.Rsquared.iloc[i])
                                    )
                                    if not MarketValueApproach or group_data.Approach.iloc[i]=="Default":
                                        Defaults = rng.binomial(
                                            n=group_data.Number_of_Exposures.iloc[i],
                                            p=ConditionalPD,
                                            size=NumberOfSimulations
                                        )
                                    else:
                                        #Determination of conditional migration probabilities
                                        Migration_prob_cond=norm.cdf(
                                            (ndtri(np.clip(np.cumsum(MigrationMatrix[int(group_data.RatingClass.iloc[i])-1,::-1])[::-1],0,1))[:, np.newaxis] - np.sqrt(group_data.Rsquared.iloc[i]) * X[group_data.Segment.iloc[i]].values[np.newaxis,:]) / np.sqrt(1 - group_data.Rsquared.iloc[i])
                                            )
                                        Migration_prob_cond = Migration_prob_cond.T
                                        Migration_prob_cond=np.column_stack([-np.diff(Migration_prob_cond, axis=1), Migration_prob_cond[:, -1]])
                                        Migrations=rng.multinomial(group_data.Number_of_Exposures.iloc[i],Migration_prob_cond)
                                        Defaults=Migrations[:,-1]
                                        PortfolioLoss_migration[:, run-1] += Migrations @ MigrationEffects[group_data.index[i],:].T/group_data.Number_of_Exposures.iloc[i]
                                else:
                                    Defaults = np.zeros(NumberOfSimulations, dtype=int)
                                    for j in range(group_data.Number_of_Exposures.iloc[i]):
                                        Return_idio=rng.normal(size=NumberOfSimulations)
                                        Return_total=math.sqrt(group_data.Rsquared.iloc[i])*X[group_data.Segment.iloc[i]].values+math.sqrt(1-group_data.Rsquared.iloc[i])*Return_idio
                                        Defaults_new=(Return_total<ndtri(group_data.PD.iloc[i])).astype(int)
                                        Defaults+=Defaults_new
                                        if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                            Migrations=Migration_events(Return_total, Migration_boundaries[int(group_data.RatingClass.iloc[i])-1,:], Defaults_new)
                                            PortfolioLoss_migration[:, run-1] += MigrationEffects[group_data.index[i],Migrations-1]/group_data.Number_of_Exposures.iloc[i]
                            else:
                                Defaults = np.ones(NumberOfSimulations, dtype=int)*group_data.Number_of_Exposures.iloc[i]
                            if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                PortfolioLoss[:, run-1] += Defaults*group_data.EAD.iloc[i]*group_data.LGD.iloc[i]/group_data.Number_of_Exposures.iloc[i]
                            elif Vasicek:
                                DefaultScenarios = np.where(Defaults > 1)[0] #Scenarios that require multiple LGDs / aggregated LGD to be modelled
                                lgd=np.zeros(NumberOfSimulations)
                                lgd_cond=np.zeros(NumberOfSimulations)
                                DefaultScenarios_Pool = np.where(Defaults > LGD_Pool_min)[0]
                                DefaultScenarios_Indiv = np.where(Defaults == 1)[0]
                                LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                # Direct simulation of LGDs for scenarios with only one default
                                if len(DefaultScenarios_Indiv) > 0:
                                    lgd[DefaultScenarios_Indiv]=fast_norm_cdf(
                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                            np.sqrt(group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios_Indiv, group_data.LGD_Segment.iloc[i]].values +
                                            np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios_Indiv]
                                        )) / np.sqrt(1 - group_data.rho.iloc[i])
                                    )
                                # Simulation of LGDs if all scenarios are subject to aggregated treatment
                                if (not LGD_Pool) or (np.min(Defaults)>LGD_Pool_min):
                                    # Calculation of conditional LGD
                                    lgd_cond[DefaultScenarios]=fast_norm_cdf(
                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values
                                        ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                    )
                                    # Calculation of conditional LGD variance
                                    ConditionalLGDvariance=1/Defaults[DefaultScenarios]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)]

                                    # Derivation of new rho parameters for aggregated LGD distribution 
                                    row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                    col_idx = np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)
                                    rho_new=RhoMapping[row_idx,col_idx]
                                
                                    # Simulation of aggregated LGD
                                    lgd[DefaultScenarios]=fast_norm_cdf(
                                            (ndtri(lgd_cond[DefaultScenarios]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios] 
                                            ) / np.sqrt(1 - rho_new)
                                        )
                                else:
                                    if len(DefaultScenarios_Pool) > 0:
                                        # Calculation of conditional LGD
                                        lgd_cond[DefaultScenarios_Pool]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios_Pool, group_data.LGD_Segment.iloc[i]].values
                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                        )
                                        # Calculation of conditional LGD variance
                                        ConditionalLGDvariance=1/Defaults[DefaultScenarios_Pool]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)]

                                        # Derivation of new rho parameters for aggregated LGD distribution 
                                        row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                        col_idx = np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)
                                        rho_new=RhoMapping[row_idx,col_idx]
                                    
                                        # Simulation of aggregated LGD
                                        lgd[DefaultScenarios_Pool]=fast_norm_cdf(
                                                (ndtri(lgd_cond[DefaultScenarios_Pool]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios_Pool] 
                                                ) / np.sqrt(1 - rho_new)
                                            )
                                    if np.min(DefaultScenarios)<=LGD_Pool_min and LGD_Pool_min>=2:
                                        for j in range(2,min(LGD_Pool_min,np.max(Defaults)+1)):
                                            DefaultScenarios_noPool = np.where(Defaults ==j)[0]
                                            if len(DefaultScenarios_noPool) > 0:
                                                idiosyncratic = rng.normal(size=(len(DefaultScenarios_noPool), j))
                                                systematic = Y.loc[DefaultScenarios_noPool, group_data.LGD_Segment.iloc[i]]
                                                latent = (np.sqrt(group_data.LGD_Rsquared.iloc[i]) * systematic.to_numpy()[:, np.newaxis] + np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * idiosyncratic)
                                                numerator = ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * latent
                                                cond_LGD_probs=fast_norm_cdf(numerator / np.sqrt(1 - group_data.rho.iloc[i]))
                                                lgd[DefaultScenarios_noPool]=np.mean(cond_LGD_probs, axis=1)
                                
                                PortfolioLoss[:, run-1] += Defaults*group_data.EAD.iloc[i]*lgd/group_data.Number_of_Exposures.iloc[i]
                            elif PLC:
                                DefaultScenarios = np.where(Defaults > 1)[0] #Scenarios that require multiple LGDs / aggregated LGD to be modelled
                                lgd=np.zeros(NumberOfSimulations)
                                lgd_cond=np.zeros(NumberOfSimulations)
                                DefaultScenarios_Pool = np.where(Defaults > LGD_Pool_min)[0]
                                DefaultScenarios_Indiv = np.where(Defaults == 1)[0]
                                LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                # Direct simulation of LGDs for scenarios with only one default
                                if len(DefaultScenarios_Indiv) > 0:
                                    lgd[DefaultScenarios_Indiv]=fast_norm_cdf(
                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                            np.sqrt(group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios_Indiv, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i]) +
                                            np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios_Indiv]
                                        )) / np.sqrt(1 - group_data.rho.iloc[i])
                                    )
                                # Simulation of LGDs if all scenarios are subject to aggregated treatment
                                if (not LGD_Pool) or (np.min(Defaults)>LGD_Pool_min):
                                    # Calculation of conditional LGD
                                    lgd_cond[DefaultScenarios]=fast_norm_cdf(
                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                        ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                    )
                                    # Calculation of conditional LGD variance
                                    ConditionalLGDvariance=1/Defaults[DefaultScenarios]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)]

                                    # Derivation of new rho parameters for aggregated LGD distribution 
                                    row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                    col_idx = np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)
                                    rho_new=RhoMapping[row_idx,col_idx]
                                
                                    # Simulation of aggregated LGD
                                    lgd[DefaultScenarios]=fast_norm_cdf(
                                            (ndtri(lgd_cond[DefaultScenarios]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios] 
                                            ) / np.sqrt(1 - rho_new)
                                        )
                                else:
                                    if len(DefaultScenarios_Pool) > 0:
                                        # Calculation of conditional LGD
                                        lgd_cond[DefaultScenarios_Pool]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios_Pool, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                        )
                                        # Calculation of conditional LGD variance
                                        ConditionalLGDvariance=1/Defaults[DefaultScenarios_Pool]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)]

                                        # Derivation of new rho parameters for aggregated LGD distribution 
                                        row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                        col_idx = np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)
                                        rho_new=RhoMapping[row_idx,col_idx]
                                    
                                        # Simulation of aggregated LGD
                                        lgd[DefaultScenarios_Pool]=fast_norm_cdf(
                                                (ndtri(lgd_cond[DefaultScenarios_Pool]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios_Pool] 
                                                ) / np.sqrt(1 - rho_new)
                                            )
                                    if np.min(DefaultScenarios)<=LGD_Pool_min and LGD_Pool_min>=2:
                                        for j in range(2,min(LGD_Pool_min,np.max(Defaults)+1)):
                                            DefaultScenarios_noPool = np.where(Defaults ==j)[0]
                                            if len(DefaultScenarios_noPool) > 0:
                                                idiosyncratic = rng.normal(size=(len(DefaultScenarios_noPool), j))
                                                systematic = (X.loc[DefaultScenarios_noPool, group_data.LGD_Segment.iloc[i]] - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                latent = (np.sqrt(group_data.LGD_Rsquared.iloc[i]) * systematic.to_numpy()[:, np.newaxis] + np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * idiosyncratic)
                                                numerator = ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * latent
                                                cond_LGD_probs=fast_norm_cdf(numerator / np.sqrt(1 - group_data.rho.iloc[i]))
                                                lgd[DefaultScenarios_noPool]=np.mean(cond_LGD_probs, axis=1)

                                PortfolioLoss[:, run-1] += Defaults*group_data.EAD.iloc[i]*lgd/group_data.Number_of_Exposures.iloc[i]
                    
                    else: # Modelling for individual clients
                        M=group_data.shape[0]
                        Return_idio=rng.normal(size=NumberOfSimulations) # Simulation of idiosyncratic returns
                        for i in range(0,M):
                            if Progress_bar_counter>0:
                                elapsed = time.time() - start_time_progress
                                est_total_time = elapsed / Progress_bar_counter * PortfolioData.shape[0]
                                eta_seconds = est_total_time - elapsed
                                eta_str = format_eta(eta_seconds)
                                Progress_bar.set(value=Progress_bar_counter,message=f"Capital window run {run}", detail=f"ETA {eta_str}") 
                            Progress_bar_counter+=1
                            if group_data.PD.iloc[i]<1:
                                Return_total=math.sqrt(group_data.Rsquared.iloc[i])*X[group_data.Segment.iloc[i]].values+math.sqrt(1-group_data.Rsquared.iloc[i])*Return_idio
                                Defaults=(Return_total<ndtri(group_data.PD.iloc[i])).astype(int)
                                if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                    Migrations=Migration_events(Return_total, Migration_boundaries[int(group_data.RatingClass.iloc[i])-1,:], Defaults)
                                    PortfolioLoss_migration[:, run-1] += MigrationEffects[group_data.index[i],Migrations-1]
                            else:
                                Defaults = np.ones(NumberOfSimulations, dtype=int)
                            if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                PortfolioLoss[:, run-1] += Defaults*group_data.EAD.iloc[i]*group_data.LGD.iloc[i]
                            elif Vasicek:
                                DefaultScenarios = np.where(Defaults > 0)[0]
                                lgd=np.zeros(NumberOfSimulations)
                                LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                if len(DefaultScenarios) > 0:
                                    lgd[DefaultScenarios]=fast_norm_cdf(
                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                            np.sqrt(group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values +
                                            np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios]
                                        )) / np.sqrt(1 - group_data.rho.iloc[i])
                                    )
                                PortfolioLoss[:, run-1] += Defaults*group_data.EAD.iloc[i]*lgd
                            elif PLC:
                                DefaultScenarios = np.where(Defaults > 0)[0]
                                lgd=np.zeros(NumberOfSimulations)
                                LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                if len(DefaultScenarios) > 0:
                                    lgd[DefaultScenarios]=fast_norm_cdf(
                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                            np.sqrt(group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i]) +
                                            np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios]
                                        )) / np.sqrt(1 - group_data.rho.iloc[i])
                                    )
                                PortfolioLoss[:, run-1] += Defaults*group_data.EAD.iloc[i]*lgd
           

        if MarketValueApproach:
            PortfolioLoss_default=PortfolioLoss[:,0:NumberOfSeeds].copy()    
            PortfolioLoss[:,0:NumberOfSeeds]+= PortfolioLoss_migration           
        
        EndTime_CW=datetime.now()  
        RunTime = EndTime_CW - StartTime_CW

        hours, remainder = divmod(RunTime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        ui.notification_show(
            f"Capital window runs finished after {int(hours)}h:{int(minutes)}m:{int(round(seconds))}s",
            type="message",
            duration=5,
        )
            
        
        if RunType2 or RunType3:
            if EScontrib:
                ES_boundary = np.zeros((NumberOfSeeds, NumberOfAllocationQuantiles))
                EScontrib_Interval = np.zeros((NumberOfSeeds, NumberOfAllocationQuantiles))
                EScontrib_Prob = np.zeros((NumberOfSeeds, NumberOfAllocationQuantiles))
                ES_scenarios = []
            if WCE:
                WCE_lower = np.zeros((NumberOfSeeds, NumberOfWCE_windows))
                WCE_upper = np.zeros((NumberOfSeeds, NumberOfWCE_windows))
                WCE_scenarios = []
                WCE_Interval = np.zeros((NumberOfSeeds, NumberOfWCE_windows))
                WCE_Prob = np.zeros((NumberOfSeeds, NumberOfWCE_windows))
        
        for run in range(1, NumberOfSeeds + 1):
            if not ImportanceSampling:
                # Sort PortfolioLoss in descending order for the current run
                PortfolioLoss_sort = np.sort(PortfolioLoss[:, run-1])[::-1]

                # Expected Loss
                EL_simulated[run-1, 0] = np.sum(PortfolioLoss[:, run-1]) / NumberOfSimulations

                # Standard Deviation
                StdDev_simulated[run-1, 0] = np.std(PortfolioLoss[:, run-1], ddof=1)

                # Value at Risk and Expected Shortfall
                for j, q in enumerate(Quantiles):
                    idx_val = NumberOfSimulations * (1 - q)
                    idx = int(1 + idx_val) if idx_val.is_integer() else int(np.ceil(idx_val))
                    idx = min(idx, NumberOfSimulations) - 1  # Ensure within bounds
                    VaR.iloc[run-1, j] = PortfolioLoss_sort[idx]
                    ES[run-1, j] = np.mean(PortfolioLoss_sort[:idx+1])

                # Economic Capital
                ECAP[run-1, :] = VaR.iloc[run-1, :] - np.sum(EL_analytical) #- (np.sum(EL_Pricing) if RunType3 else 0)

            else:
                # With importance sampling
                weights = PortfolioLoss[:, NumberOfSeeds + run-1]
                sorted_indices = np.argsort(PortfolioLoss[:, run-1])[::-1]
                PortfolioLoss_sorted = PortfolioLoss[sorted_indices, run-1]
                weights_sorted = weights[sorted_indices]

                PortfolioLoss_sort = np.column_stack((PortfolioLoss_sorted, weights_sorted))
                LikelihoodRatio_cum = np.cumsum(PortfolioLoss_sort[:, 1])

                EL_simulated[run-1, 0] = np.sum(PortfolioLoss[:, run-1] * weights)
                StdDev_simulated[run-1, 0] = np.sqrt(np.sum(weights * (PortfolioLoss[:, run-1] - EL_simulated[run-1, 0])**2))

                for j, q in enumerate(Quantiles):
                    cum_diff = np.abs(1 - LikelihoodRatio_cum - q)
                    if np.min(cum_diff) == 0:
                        VaR_index = np.where(cum_diff == 0)[0][0] + 1
                    else:
                        VaR_index = np.argmax(1 / (LikelihoodRatio_cum - (1 - q)))
                    VaR_index = min(VaR_index, NumberOfSimulations - 1)

                    VaR.iloc[run-1, j] = PortfolioLoss_sort[VaR_index, 0]
                    ES[run-1, j] = np.sum(PortfolioLoss_sort[:VaR_index + 1, 0] * PortfolioLoss_sort[:VaR_index + 1, 1]) / \
                                np.sum(PortfolioLoss_sort[:VaR_index + 1, 1])

                ECAP[run-1, :] = VaR.iloc[run-1, :] - np.sum(EL_analytical) #- (np.sum(EL_Pricing) if RunType3 else 0)
            if MarketValueApproach:
                #Output for defaults only
                if not ImportanceSampling:
                    # Sort PortfolioLoss in descending order for the current run
                    PortfolioLoss_default_sort = np.sort(PortfolioLoss_default[:, run-1])[::-1]

                    # Expected Loss
                    EL_simulated_default[run-1, 0] = np.sum(PortfolioLoss_default[:, run-1]) / NumberOfSimulations

                    # Standard Deviation
                    StdDev_simulated_default[run-1, 0] = np.std(PortfolioLoss_default[:, run-1], ddof=1)

                    # Value at Risk and Expected Shortfall
                    for j, q in enumerate(Quantiles):
                        idx_val = NumberOfSimulations * (1 - q)
                        idx = int(1 + idx_val) if idx_val.is_integer() else int(np.ceil(idx_val))
                        idx = min(idx, NumberOfSimulations) - 1  # Ensure within bounds
                        VaR_default.iloc[run-1, j] = PortfolioLoss_default_sort[idx]
                        ES_default[run-1, j] = np.mean(PortfolioLoss_default_sort[:idx+1])

                    # Economic Capital
                    ECAP_default[run-1, :] = VaR_default.iloc[run-1, :] - np.sum(EL_analytical_default) #- (np.sum(EL_Pricing) if RunType3 else 0)

                else:
                    # With importance sampling
                    weights = PortfolioLoss[:, NumberOfSeeds + run-1]
                    sorted_indices = np.argsort(PortfolioLoss_default[:, run-1])[::-1]
                    PortfolioLoss_default_sorted = PortfolioLoss_default[sorted_indices, run-1]
                    weights_sorted = weights[sorted_indices]

                    PortfolioLoss_default_sort = np.column_stack((PortfolioLoss_default_sorted, weights_sorted))
                    LikelihoodRatio_cum = np.cumsum(PortfolioLoss_default_sort[:, 1])

                    EL_simulated_default[run-1, 0] = np.sum(PortfolioLoss_default[:, run-1] * weights)
                    StdDev_simulated_default[run-1, 0] = np.sqrt(np.sum(weights * (PortfolioLoss_default[:, run-1] - EL_simulated_default[run-1, 0])**2))

                    for j, q in enumerate(Quantiles):
                        cum_diff = np.abs(1 - LikelihoodRatio_cum - q)
                        if np.min(cum_diff) == 0:
                            VaR_index = np.where(cum_diff == 0)[0][0] + 1
                        else:
                            VaR_index = np.argmax(1 / (LikelihoodRatio_cum - (1 - q)))
                        VaR_index = min(VaR_index, NumberOfSimulations - 1)

                        VaR_default.iloc[run-1, j] = PortfolioLoss_default_sort[VaR_index, 0]
                        ES_default[run-1, j] = np.sum(PortfolioLoss_default_sort[:VaR_index + 1, 0] * PortfolioLoss_default_sort[:VaR_index + 1, 1]) / \
                                    np.sum(PortfolioLoss_default_sort[:VaR_index + 1, 1])

                    ECAP_default[run-1, :] = VaR_default.iloc[run-1, :] - np.sum(EL_analytical_default) #- (np.sum(EL_Pricing) if RunType3 else 0)

                #Output for migration only
                if not ImportanceSampling:
                    # Sort PortfolioLoss in descending order for the current run
                    PortfolioLoss_migration_sort = np.sort(PortfolioLoss_migration[:, run-1])[::-1]

                    # Expected Loss
                    EL_simulated_migration[run-1, 0] = np.sum(PortfolioLoss_migration[:, run-1]) / NumberOfSimulations

                    # Standard Deviation
                    StdDev_simulated_migration[run-1, 0] = np.std(PortfolioLoss_migration[:, run-1], ddof=1)

                    # Value at Risk and Expected Shortfall
                    for j, q in enumerate(Quantiles):
                        idx_val = NumberOfSimulations * (1 - q)
                        idx = int(1 + idx_val) if idx_val.is_integer() else int(np.ceil(idx_val))
                        idx = min(idx, NumberOfSimulations) - 1  # Ensure within bounds
                        VaR_migration.iloc[run-1, j] = PortfolioLoss_migration_sort[idx]
                        ES_migration[run-1, j] = np.mean(PortfolioLoss_migration_sort[:idx+1])

                    # Economic Capital
                    ECAP_migration[run-1, :] = VaR_migration.iloc[run-1, :] - np.sum(EL_analytical_migration) #- (np.sum(EL_Pricing) if RunType3 else 0)

                else:
                    # With importance sampling
                    weights = PortfolioLoss[:, NumberOfSeeds + run-1]
                    sorted_indices = np.argsort(PortfolioLoss_migration[:, run-1])[::-1]
                    PortfolioLoss_migration_sorted = PortfolioLoss_migration[sorted_indices, run-1]
                    weights_sorted = weights[sorted_indices]

                    PortfolioLoss_migration_sort = np.column_stack((PortfolioLoss_migration_sorted, weights_sorted))
                    LikelihoodRatio_cum = np.cumsum(PortfolioLoss_migration_sort[:, 1])

                    EL_simulated_migration[run-1, 0] = np.sum(PortfolioLoss_migration[:, run-1] * weights)
                    StdDev_simulated_migration[run-1, 0] = np.sqrt(np.sum(weights * (PortfolioLoss_migration[:, run-1] - EL_simulated_migration[run-1, 0])**2))

                    for j, q in enumerate(Quantiles):
                        cum_diff = np.abs(1 - LikelihoodRatio_cum - q)
                        if np.min(cum_diff) == 0:
                            VaR_index = np.where(cum_diff == 0)[0][0] + 1
                        else:
                            VaR_index = np.argmax(1 / (LikelihoodRatio_cum - (1 - q)))
                        VaR_index = min(VaR_index, NumberOfSimulations - 1)

                        VaR_migration.iloc[run-1, j] = PortfolioLoss_migration_sort[VaR_index, 0]
                        ES_migration[run-1, j] = np.sum(PortfolioLoss_migration_sort[:VaR_index + 1, 0] * PortfolioLoss_migration_sort[:VaR_index + 1, 1]) / \
                                    np.sum(PortfolioLoss_migration_sort[:VaR_index + 1, 1])

                    ECAP_migration[run-1, :] = VaR_migration.iloc[run-1, :] - np.sum(EL_analytical_migration) #- (np.sum(EL_Pricing) if RunType3 else 0)


            # Identification of relevant scenarios for capital allocation
            if RunType2 or RunType3:
                if EScontrib:
                    if not ImportanceSampling:
                        PortfolioLoss_cum = np.cumsum(PortfolioLoss_sort) / np.arange(1, len(PortfolioLoss_sort) + 1)
                    else:
                        PortfolioLoss_cum = np.cumsum(PortfolioLoss_sort[:, 0] * PortfolioLoss_sort[:, 1]) / LikelihoodRatio_cum
                    
                    for j in range(NumberOfAllocationQuantiles):
                        if not ESLowerBoundary:
                            if ESType1:
                                idx = np.argmin(np.abs(PortfolioLoss_cum - VaR.iloc[run-1, j]))
                                ES_boundary[run-1, j] = PortfolioLoss_sort[idx, 0] if ImportanceSampling else PortfolioLoss_sort[idx]
                            else:
                                ES_boundary[run-1, j] = VaR.iloc[run-1, j]
                        else:
                            ES_boundary[run-1, j] = LowerBoundary_ES

                        #if not ParallelComp or ParallelType1:
                        scenario = np.where(PortfolioLoss[:, run-1] >= ES_boundary[run-1, j])[0]
                        ES_scenarios.append(scenario)
                        EScontrib_Interval[run-1, j] = len(scenario)
                        """ else:
                            for batch in range(BatchSize_Sim):
                                start = batch * NumberOfSimulations_Batch
                                end = start + NumberOfSimulations_Batch
                                scenario = np.where(PortfolioLoss[start:end, run] >= ES_boundary[run, j])[0] + start
                                ES_scenarios.append(scenario)
                                EScontrib_Interval[run, j] += len(scenario) """

                        if ImportanceSampling:
                            #if not ParallelComp or ParallelType1:
                            EScontrib_Prob[run-1, j] = np.sum(PortfolioLoss[scenario, NumberOfSeeds + run-1])
                            """ else:
                                all_batches = [
                                    s for s in ES_scenarios[(run * NumberOfAllocationQuantiles * BatchSize_Sim + j * BatchSize_Sim):(run * NumberOfAllocationQuantiles * BatchSize_Sim + (j+1) * BatchSize_Sim)]
                                ]
                                combined = np.concatenate(all_batches)
                                EScontrib_Prob[run, j] = np.sum(PortfolioLoss[combined, NumberOfSeeds + run]) """
                
                if WCE:
                    for j in range(NumberOfWCE_windows):
                        if not ImportanceSampling:
                            WCE_lower[run-1, j] = np.quantile(PortfolioLoss[:, run-1], max(0, LowerBoundary[j]))
                            WCE_upper[run-1, j] = np.quantile(PortfolioLoss[:, run-1], min(1, UpperBoundary[j]))
                        else:
                            idx_lower = np.argmin(np.abs(LikelihoodRatio_cum - (1 - max(0, LowerBoundary[j]))))
                            idx_upper = np.argmin(np.abs(LikelihoodRatio_cum - (1 - min(1, UpperBoundary[j]))))
                            WCE_lower[run-1, j] = PortfolioLoss_sort[idx_lower, 0]
                            WCE_upper[run-1, j] = PortfolioLoss_sort[idx_upper, 0]

                        #if not ParallelComp or ParallelType1:
                        mask = (PortfolioLoss[:, run-1] >= WCE_lower[run-1, j]) & (PortfolioLoss[:, run-1] <= WCE_upper[run-1, j])
                        scenario = np.where(mask)[0]
                        WCE_scenarios.append(scenario)
                        WCE_Interval[run-1, j] = len(scenario)
                        """   else:
                            for batch in range(BatchSize_Sim):
                                start = batch * NumberOfSimulations_Batch
                                end = start + NumberOfSimulations_Batch
                                mask = (PortfolioLoss[start:end, run] >= WCE_lower[run, j]) & (PortfolioLoss[start:end, run] <= WCE_upper[run, j])
                                scenario = np.where(mask)[0] + start
                                WCE_scenarios.append(scenario)
                                WCE_Interval[run, j] += len(scenario) """

                        if ImportanceSampling:
                            #if not ParallelComp or ParallelType1:
                            WCE_Prob[run-1, j] = np.sum(PortfolioLoss[scenario, NumberOfSeeds + run-1])
                            """ else:
                                all_batches = [
                                    s for s in WCE_scenarios[(run * NumberOfWCE_windows * BatchSize_Sim + j * BatchSize_Sim):(run * NumberOfWCE_windows * BatchSize_Sim + (j+1) * BatchSize_Sim)]
                                ]
                                combined = np.concatenate(all_batches)
                                WCE_Prob[run, j] = np.sum(PortfolioLoss[combined, NumberOfSeeds + run]) """

      
        
        #########################
        ### Output generation ###
        #########################

        
        # Combine all results horizontally
        PortfolioResults = np.hstack([
            EL_simulated,                            # shape (NumberOfSeeds, 1)
            StdDev_simulated,                        # shape (NumberOfSeeds, 1)
            VaR,                                     # shape (NumberOfSeeds, NumberOfCIs)
            ECAP,                                    # shape (NumberOfSeeds, NumberOfCIs)
            ES                                       # shape (NumberOfSeeds, NumberOfCIs)
        ])

        # Compute average row
        average_row = np.mean(PortfolioResults, axis=0, keepdims=True)

        # Stack the average as the last row
        PortfolioResults = np.vstack([PortfolioResults, average_row])

        row_names = [f"Run {i+1}" for i in range(NumberOfSeeds)] + ["Average"]
        column_names = (
            ["Expected Loss", "Standard Deviation"] +
            [f"VaR ({(q*100)}%)" for q in Quantiles] +
            [f"ECAP ({(q*100)}%)" for q in Quantiles] +
            [f"Expected Shortfall ({(q*100)}%)" for q in Quantiles]
        )

        PortfolioResults = pd.DataFrame(PortfolioResults, index=row_names, columns=column_names)

        # Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
        file_name_PortfolioResults = f"Portfolio_Results_{timestamp}.csv"
        
        @render.download(filename=file_name_PortfolioResults)
        def downloadPortfolioResults():
            yield PortfolioResults.to_csv() 
        PortfolioResults.to_csv(file_name_PortfolioResults, index=True)

        # Create summary table for run results
        @render.ui
        def PortfolioResults_summary():
            df = PortfolioResults.copy()
            df.index.name = "Run"
            df = df.reset_index()

            def row_to_html(row, bold=False):
                style = "font-weight:bold;" if bold else ""
                cells = "".join(
                    f"<td style='text-align:right; {style}'>{int(value):,}</td>"
                    if isinstance(value, (int, float))
                    else f"<td style='{style}'>{value}</td>"
                    for value in row
                )
                return f"<tr>{cells}</tr>"

            # Build HTML table manually
            headers = "".join(f"<th>{col}</th>" for col in df.columns)
            rows = [
                row_to_html(df.iloc[i], bold=(i == len(df) - 1))
                for i in range(len(df))
            ]

            table_html = f"""
            <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            """
            return ui.HTML(table_html)
        
        ######################################################################
        ### Output generation for standalone default and migration effects ###
        ######################################################################

        if MarketValueApproach:
            # Combine all results horizontally
            PortfolioResults_default = np.hstack([
                EL_simulated_default,                            # shape (NumberOfSeeds, 1)
                StdDev_simulated_default,                        # shape (NumberOfSeeds, 1)
                VaR_default,                                     # shape (NumberOfSeeds, NumberOfCIs)
                ECAP_default,                                    # shape (NumberOfSeeds, NumberOfCIs)
                ES_default                                       # shape (NumberOfSeeds, NumberOfCIs)
            ])

            # Compute average row
            average_row = np.mean(PortfolioResults_default, axis=0, keepdims=True)

            # Stack the average as the last row
            PortfolioResults_default = np.vstack([PortfolioResults_default, average_row])

            row_names = [f"Run {i+1}" for i in range(NumberOfSeeds)] + ["Average"]
            column_names = (
                ["Expected Loss", "Standard Deviation"] +
                [f"VaR ({(q*100)}%)" for q in Quantiles] +
                [f"ECAP ({(q*100)}%)" for q in Quantiles] +
                [f"Expected Shortfall ({(q*100)}%)" for q in Quantiles]
            )

            PortfolioResults_default = pd.DataFrame(PortfolioResults_default, index=row_names, columns=column_names)

            # Save to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
            file_name_PortfolioResults_default = f"Portfolio_Results_default_{timestamp}.csv"
            
            @render.download(filename=file_name_PortfolioResults_default)
            def downloadPortfolioResults_default():
                yield PortfolioResults_default.to_csv() 
            PortfolioResults_default.to_csv(file_name_PortfolioResults_default, index=True)

            # Create summary table for run results
            @render.ui
            def PortfolioResults_summary_default():
                df = PortfolioResults_default.copy()
                df.index.name = "Run"
                df = df.reset_index()

                def row_to_html(row, bold=False):
                    style = "font-weight:bold;" if bold else ""
                    cells = "".join(
                        f"<td style='text-align:right; {style}'>{int(value):,}</td>"
                        if isinstance(value, (int, float))
                        else f"<td style='{style}'>{value}</td>"
                        for value in row
                    )
                    return f"<tr>{cells}</tr>"

                # Build HTML table manually
                headers = "".join(f"<th>{col}</th>" for col in df.columns)
                rows = [
                    row_to_html(df.iloc[i], bold=(i == len(df) - 1))
                    for i in range(len(df))
                ]

                table_html = f"""
                <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                    <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                        <thead><tr>{headers}</tr></thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
                """
                return ui.HTML(table_html)
            
            # Combine all results horizontally
            PortfolioResults_migration = np.hstack([
                EL_simulated_migration,                            # shape (NumberOfSeeds, 1)
                StdDev_simulated_migration,                        # shape (NumberOfSeeds, 1)
                VaR_migration,                                     # shape (NumberOfSeeds, NumberOfCIs)
                ECAP_migration,                                    # shape (NumberOfSeeds, NumberOfCIs)
                ES_migration                                       # shape (NumberOfSeeds, NumberOfCIs)
            ])

            # Compute average row
            average_row = np.mean(PortfolioResults_migration, axis=0, keepdims=True)

            # Stack the average as the last row
            PortfolioResults_migration = np.vstack([PortfolioResults_migration, average_row])

            row_names = [f"Run {i+1}" for i in range(NumberOfSeeds)] + ["Average"]
            column_names = (
                ["Expected Loss", "Standard Deviation"] +
                [f"VaR ({(q*100)}%)" for q in Quantiles] +
                [f"ECAP ({(q*100)}%)" for q in Quantiles] +
                [f"Expected Shortfall ({(q*100)}%)" for q in Quantiles]
            )

            PortfolioResults_migration = pd.DataFrame(PortfolioResults_migration, index=row_names, columns=column_names)

            # Save to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
            file_name_PortfolioResults_migration = f"Portfolio_Results_migration_{timestamp}.csv"
            
            @render.download(filename=file_name_PortfolioResults_migration)
            def downloadPortfolioResults_migration():
                yield PortfolioResults_migration.to_csv() 
            PortfolioResults_migration.to_csv(file_name_PortfolioResults_migration, index=True)

            # Create summary table for run results
            @render.ui
            def PortfolioResults_summary_migration():
                df = PortfolioResults_migration.copy()
                df.index.name = "Run"
                df = df.reset_index()

                def row_to_html(row, bold=False):
                    style = "font-weight:bold;" if bold else ""
                    cells = "".join(
                        f"<td style='text-align:right; {style}'>{int(value):,}</td>"
                        if isinstance(value, (int, float))
                        else f"<td style='{style}'>{value}</td>"
                        for value in row
                    )
                    return f"<tr>{cells}</tr>"

                # Build HTML table manually
                headers = "".join(f"<th>{col}</th>" for col in df.columns)
                rows = [
                    row_to_html(df.iloc[i], bold=(i == len(df) - 1))
                    for i in range(len(df))
                ]

                table_html = f"""
                <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                    <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                        <thead><tr>{headers}</tr></thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
                """
                return ui.HTML(table_html)


        # Generate histogram images and store base64 strings
        images = []
        for run in range(NumberOfSeeds+1):

            plot_loss_distribution(
                run,
                PortfolioLoss,
                VaR,  # DataFrame: rows = seeds, cols = quantiles
                ES,   # NumPy array
                ECAP, # NumPy array
                EL_simulated,  # NumPy array
                StdDev_simulated,  # NumPy array
                Quantiles,  # list or array of quantiles
                NumberOfSeeds,
                ImportanceSampling,
                MarketValueApproach,
                ax=None
            )
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=600)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            images.append(img_str)
            buf.close()
            plt.close()

        histogram_images.set(images)  # Store the image strings

        #Generate pdf to download loss distributions
        if GraphicalOutput:
            filename_LossDistributions=lambda: f"LossDistributions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
            @render.download(filename=filename_LossDistributions)
            def downloadLossDist_graph():
                pdf_bytes = generate_pdf_from_base64_images(images)
                yield pdf_bytes
        
        # Generate tabular output of loss distribution

        # Create quantile vector
        LossDist_quantiles = np.concatenate([
            np.arange(1, 76) / 100,
            0.75 + np.arange(1, 151) / 1000,
            0.9 + np.arange(1, 1000) / 10000
        ])

        LossOutput = np.zeros((len(LossDist_quantiles), NumberOfSeeds))

        if not ImportanceSampling:
            for run in range(1, NumberOfSeeds + 1):
                # Sort portfolio loss in descending order
                PortfolioLoss_sort = np.sort(PortfolioLoss[:, run-1])[::-1]

                indices = NumberOfSimulations * (1 - LossDist_quantiles)
                is_whole = np.isclose(indices % 1, 0)
                idx = np.where(is_whole, np.round(1 + indices), np.ceil(indices)).astype(int)

                # Make sure indices are within bounds
                idx = np.clip(idx - 1, 0, NumberOfSimulations - 1)  # subtract 1 for 0-based indexing

                LossOutput[:, run-1] = PortfolioLoss_sort[idx]
        else:
            for run in range(1, NumberOfSeeds + 1):
                 # Extract losses and weights for this run
                losses = PortfolioLoss[:, run-1]
                weights = PortfolioLoss[:, NumberOfSeeds + run-1]

                # Sort losses and apply the same order to weights
                sort_idx = np.argsort(-losses)  # Descending
                losses_sorted = losses[sort_idx]
                weights_sorted = weights[sort_idx]

                # Cumulative weights
                LikelihoodRatio_cum = np.cumsum(weights_sorted)

                for k, q in enumerate(LossDist_quantiles):
                    threshold = 1 - q

                    if threshold > np.max(LikelihoodRatio_cum):
                        LossOutput[k, run-1] = np.min(losses_sorted)
                    else:
                        diffs = (LikelihoodRatio_cum - threshold)
                        if np.min(np.abs(diffs)) == 0:
                            VaR_index = np.argmin(diffs) + 1
                        else:
                            # Avoid division by zero or instability
                            adjusted_diffs = np.where(diffs == 0, np.inf, 1 / diffs)
                            VaR_index = np.argmax(adjusted_diffs)
                        VaR_index = np.clip(VaR_index, 0, NumberOfSimulations - 1)
                        LossOutput[k, run-1] = losses_sorted[VaR_index]

        # Create DataFrame for better labeling
        LossOutput = pd.DataFrame(LossOutput, 
                                    columns=[f"Run {i+1}" for i in range(NumberOfSeeds)],
                                    index=LossDist_quantiles)

        # Add the Average column
        LossOutput["Average"] = LossOutput.mean(axis=1)

        @render.ui
        def LossDistribution_quantiles():
            df = LossOutput.copy()
            df.index.name = "Quantile"
            df.index = df.index.map(lambda x: f"{x * 100:.2f}%")
            df = df.reset_index()

            def row_to_html(row, bold=False):
                style = "font-weight:bold;" if bold else ""
                cells = "".join(
                    f"<td style='text-align:center; {style}'>{int(value):,}</td>"
                    if isinstance(value, (int, float))
                    else f"<td style='{style}'>{value}</td>"
                    for value in row
                )
                return f"<tr>{cells}</tr>"

            # Build HTML table manually
            headers = "".join(
                f"<th style='text-align:center'>{col}</th>" if i != 0 else f"<th>{col}</th>"
                for i, col in enumerate(df.columns)
            )
            rows = [
                row_to_html(df.iloc[i], bold=False)
                for i in range(len(df))
            ]

            table_html = f"""
            <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                    <thead style="position: sticky; top: 0; background-color: white; z-index: 1;">
                        <tr>{headers}</tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            """
            return ui.HTML(table_html)
                            
        if LossDistribution: 
            timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
            file_name_LossOutput = f"Quantile_Loss_Distribution_{timestamp}.csv"
                
            @render.download(filename=file_name_LossOutput)
            def downloadLossDist_quantiles():
                yield LossOutput.to_csv() 

        # Output of full loss distribution 
        if LossDistribution2:
            LossOutput2 = np.zeros((NumberOfSimulations, NumberOfSeeds * (1 + int(ImportanceSampling))))

            if not ImportanceSampling:
                for run in range(NumberOfSeeds):
                    LossOutput2[:, run] = np.sort(PortfolioLoss[:, run])[::-1]  # descending sort
                column_names = [f"Run {i+1}" for i in range(NumberOfSeeds)]

            else:
                for run in range(NumberOfSeeds):
                    sorted_losses = np.sort(PortfolioLoss[:, run])[::-1]
                    weights = PortfolioLoss[np.argsort(PortfolioLoss[:, run])[::-1], NumberOfSeeds + run]
                    LossOutput2[:, run * 2] = sorted_losses
                    LossOutput2[:, run * 2 + 1] = weights
                column_names = []
                for i in range(NumberOfSeeds):
                    column_names.extend([f"Run {i+1}", f"IS weight run {i+1}"])

            # Optional: create a DataFrame
            LossOutput2 = pd.DataFrame(LossOutput2, columns=column_names)

            timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
            file_name_LossOutput2 = f"Full_Loss_Distribution_{timestamp}.csv"
                
            @render.download(filename=file_name_LossOutput2)
            def downloadLossDist_full():
                yield LossOutput2.to_csv() 





        ########################################
        ### Capital allocation runs          ###
        ########################################

        pr = cProfile.Profile()
        pr.enable()


   

        if RunType2 or RunType3:
            ui.notification_show("Capital allocation runs started")
            StartTime_CA=datetime.now()  
            ui.notification_show(
                f"Start capital allolcation runs at {StartTime_CA.strftime('%H:%M:%S')} ...",
                type="message",  # Could also be "default", "warning", "error"
                duration=5
            )    

            # Simulated expected losses per cluster
            ExpLoss_simulated = np.zeros((NumberOfPools, NumberOfSeeds + 1))

            if EScontrib:
                if EScontrib_sim:
                    # Expected shortfall contributions (simulated)
                    EScontrib_allocation = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))
                    if CondPDLGD:
                        EScontrib_PD = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))
                        EScontrib_LGD = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))
                
                if EScontrib_analytic:
                    # Expected shortfall contributions (analytic)
                    EScontrib_allocation_analytic = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))
                    if CondPDLGD:
                        EScontrib_PD_analytic = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))
                        EScontrib_LGD_analytic = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))

            if MRcontribSim:
                MRcontribSim_allocation = np.zeros((NumberOfPools, NumberOfAllocationQuantiles * (NumberOfSeeds + 1)))

            if WCE:
                if WCE_sim:
                    WCE_allocation = np.zeros((NumberOfPools, NumberOfWCE_windows * (NumberOfSeeds + 1)))
                    if CondPDLGD:
                        WCE_PD = np.zeros((NumberOfPools, NumberOfWCE_windows * (NumberOfSeeds + 1)))
                        WCE_LGD = np.zeros((NumberOfPools, NumberOfWCE_windows * (NumberOfSeeds + 1)))

                if WCE_analytic:
                    WCE_allocation_analytic = np.zeros((NumberOfPools, NumberOfWCE_windows * (NumberOfSeeds + 1)))
                    if CondPDLGD:
                        WCE_PD_analytic = np.zeros((NumberOfPools, NumberOfWCE_windows * (NumberOfSeeds + 1)))
                        WCE_LGD_analytic = np.zeros((NumberOfPools, NumberOfWCE_windows * (NumberOfSeeds + 1)))

        
            for run in range(1, NumberOfSeeds + 1):
                ui.notification_show(f"Capital allocation run {run} started") 
                rng = np.random.default_rng(seed=Rand_seed+run)

                Progress_bar_counter=0
                with ui.Progress(min=1, max=PortfolioData.shape[0]) as Progress_bar:
                    Progress_bar.set(message=f"Capital allocation run {run} in progress") 

                    #Generate correlated systematic factors
                    if not AntitheticSampling:
                        X = rng.normal(size=(NumberOfSimulations, NumberOfFactors))
                    else:
                        X_half = rng.normal(size=(int(NumberOfSimulations / 2), NumberOfFactors))
                        X = np.vstack((X_half, -X_half))
                        del X_half

                    if ImportanceSampling:
                        if ISType1:
                            X[:, 0] *= ScalingFactor  # Application of importance sampling on first factor
                        if ISType2:
                            for k in range(NumberOfFactors):
                                X[:, k] += Mean_shift[k]  # Application of mean shift
                                        
                    X=X @ FactorTable.T # Derivation of correlated systematic factor returns

                    #Generate correlated systematic LGD factors (if PLC is not enabled)
                    if Vasicek:
                        if not AntitheticSampling:
                            Y = rng.normal(size=(NumberOfSimulations, NumberOfLGDFactors))
                        else:
                            Y_half = rng.normal(size=(int(NumberOfSimulations / 2), NumberOfLGDFactors))
                            Y = np.vstack((Y_half, -Y_half))
                            del Y_half

                        Y=Y @ LGDFactorTable.T # Derivation of correlated systematic LGD factor returns

                
                    if RunType3:
                        rng = np.random.default_rng(seed=Rand_seed+run+27644437) # To control random seed for pricing run

                    start_time_progress = time.time()
                    for group_id in GroupIDs:
                        # Filter rows where GroupID equals current group_id
                        group_data = PortfolioData[PortfolioData['GroupID'] == group_id]
                        
                        # Get max of Number_of_Exposures in that group
                        max_exposures = group_data['Number_of_Exposures'].max()
            
                        if max_exposures > 1: # Modelling for exposure pools
                            M=group_data.shape[0]
                            for i in range(0,M):
                                if Progress_bar_counter>0:
                                    elapsed = time.time() - start_time_progress
                                    est_total_time = elapsed / Progress_bar_counter * PortfolioData.shape[0]
                                    eta_seconds = est_total_time - elapsed
                                    eta_str = format_eta(eta_seconds)
                                    Progress_bar.set(value=Progress_bar_counter,message=f"Capital allocation run {run}", detail=f"ETA {eta_str}")
                                Progress_bar_counter+=1 
                                if group_data.PD.iloc[i]<1:
                                    ConditionalPD = fast_norm_cdf(
                                        (ndtri(group_data.PD.iloc[i]) - np.sqrt(group_data.Rsquared.iloc[i]) * X[group_data.Segment.iloc[i]].values) / np.sqrt(1 - group_data.Rsquared.iloc[i])
                                    )
                                    if group_data.Number_of_Exposures.iloc[i] > MaxPoolSize:
                                        if not MarketValueApproach or group_data.Approach.iloc[i]=="Default":
                                            Defaults = rng.binomial(
                                                n=group_data.Number_of_Exposures.iloc[i],
                                                p=ConditionalPD,
                                                size=NumberOfSimulations
                                            )
                                        else:
                                            #Determination of conditional migration probabilities
                                            Migration_prob_cond=norm.cdf(
                                                (ndtri(np.clip(np.cumsum(MigrationMatrix[int(group_data.RatingClass.iloc[i])-1,::-1])[::-1],0,1))[:, np.newaxis] - np.sqrt(group_data.Rsquared.iloc[i]) * X[group_data.Segment.iloc[i]].values[np.newaxis,:]) / np.sqrt(1 - group_data.Rsquared.iloc[i])
                                                )
                                            Migration_prob_cond = Migration_prob_cond.T
                                            Migration_prob_cond=np.column_stack([-np.diff(Migration_prob_cond, axis=1), Migration_prob_cond[:, -1]])
                                            Migrations=rng.multinomial(group_data.Number_of_Exposures.iloc[i],Migration_prob_cond)
                                            Defaults=Migrations[:,-1]
                                            PoolLoss_migration = Migrations @ MigrationEffects[group_data.index[i],:].T/group_data.Number_of_Exposures.iloc[i]
                                    else:
                                        ConditionalPD = np.ones(NumberOfSimulations)
                                        Defaults = np.zeros(NumberOfSimulations, dtype=int)
                                        if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                            PoolLoss_migration = np.zeros(NumberOfSimulations)
                                        for j in range(group_data.Number_of_Exposures.iloc[i]):
                                            Return_idio=rng.normal(size=NumberOfSimulations)
                                            Return_total=math.sqrt(group_data.Rsquared.iloc[i])*X[group_data.Segment.iloc[i]].values+math.sqrt(1-group_data.Rsquared.iloc[i])*Return_idio
                                            Defaults_new=(Return_total<ndtri(group_data.PD.iloc[i])).astype(int)
                                            Defaults+=Defaults_new
                                            if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                                Migrations=Migration_events(Return_total, Migration_boundaries[int(group_data.RatingClass.iloc[i])-1,:], Defaults_new)
                                                PoolLoss_migration += MigrationEffects[group_data.index[i],Migrations-1]/group_data.Number_of_Exposures.iloc[i]
                                else:
                                    ConditionalPD = np.ones(NumberOfSimulations)
                                    Defaults = np.ones(NumberOfSimulations, dtype=int)*group_data.Number_of_Exposures.iloc[i]
                                    if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                        PoolLoss_migration=np.zeros(NumberOfSimulations) #Defaulted clients cannot have migration effects
                                if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                    PoolLoss = Defaults*group_data.EAD.iloc[i]*group_data.LGD.iloc[i]/group_data.Number_of_Exposures.iloc[i]
                                elif Vasicek:
                                    DefaultScenarios = np.where(Defaults > 1)[0] #Scenarios that require multiple LGDs / aggregated LGD to be modelled
                                    lgd=np.zeros(NumberOfSimulations)
                                    lgd_cond=np.zeros(NumberOfSimulations)
                                    DefaultScenarios_Pool = np.where(Defaults > LGD_Pool_min)[0]
                                    DefaultScenarios_Indiv = np.where(Defaults == 1)[0]
                                    LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                    # Direct simulation of LGDs for scenarios with only one default
                                    if len(DefaultScenarios_Indiv) > 0:
                                        lgd[DefaultScenarios_Indiv]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                                np.sqrt(group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios_Indiv, group_data.LGD_Segment.iloc[i]].values +
                                                np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios_Indiv]
                                            )) / np.sqrt(1 - group_data.rho.iloc[i])
                                        )
                                    # Simulation of LGDs if all scenarios are subject to aggregated treatment
                                    if (not LGD_Pool) or (np.min(Defaults)>LGD_Pool_min):
                                        # Calculation of conditional LGD
                                        lgd_cond[DefaultScenarios]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values
                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                        )
                                        # Calculation of conditional LGD variance
                                        ConditionalLGDvariance=1/Defaults[DefaultScenarios]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)]

                                        # Derivation of new rho parameters for aggregated LGD distribution 
                                        row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                        col_idx = np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)
                                        rho_new=RhoMapping[row_idx,col_idx]
                                    
                                        # Simulation of aggregated LGD
                                        lgd[DefaultScenarios]=fast_norm_cdf(
                                                (ndtri(lgd_cond[DefaultScenarios]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios] 
                                                ) / np.sqrt(1 - rho_new)
                                            )
                                    else:
                                        if len(DefaultScenarios_Pool) > 0:
                                            # Calculation of conditional LGD
                                            lgd_cond[DefaultScenarios_Pool]=fast_norm_cdf(
                                                (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios_Pool, group_data.LGD_Segment.iloc[i]].values
                                                ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                            )
                                            # Calculation of conditional LGD variance
                                            ConditionalLGDvariance=1/Defaults[DefaultScenarios_Pool]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)]

                                            # Derivation of new rho parameters for aggregated LGD distribution 
                                            row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                            col_idx = np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)
                                            rho_new=RhoMapping[row_idx,col_idx]
                                        
                                            # Simulation of aggregated LGD
                                            lgd[DefaultScenarios_Pool]=fast_norm_cdf(
                                                    (ndtri(lgd_cond[DefaultScenarios_Pool]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios_Pool] 
                                                    ) / np.sqrt(1 - rho_new)
                                                )
                                        if np.min(DefaultScenarios)<=LGD_Pool_min and LGD_Pool_min>=2:
                                            for j in range(2,min(LGD_Pool_min,np.max(Defaults)+1)):
                                                DefaultScenarios_noPool = np.where(Defaults ==j)[0]
                                                if len(DefaultScenarios_noPool) > 0:
                                                    idiosyncratic = rng.normal(size=(len(DefaultScenarios_noPool), j))
                                                    systematic = Y.loc[DefaultScenarios_noPool, group_data.LGD_Segment.iloc[i]]
                                                    latent = (np.sqrt(group_data.LGD_Rsquared.iloc[i]) * systematic.to_numpy()[:, np.newaxis] + np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * idiosyncratic)
                                                    numerator = ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * latent
                                                    cond_LGD_probs=fast_norm_cdf(numerator / np.sqrt(1 - group_data.rho.iloc[i]))
                                                    lgd[DefaultScenarios_noPool]=np.mean(cond_LGD_probs, axis=1)
                                    
                                    PoolLoss = Defaults*group_data.EAD.iloc[i]*lgd/group_data.Number_of_Exposures.iloc[i]
                                elif PLC:
                                    DefaultScenarios = np.where(Defaults > 1)[0] #Scenarios that require multiple LGDs / aggregated LGD to be modelled
                                    lgd=np.zeros(NumberOfSimulations)
                                    lgd_cond=np.zeros(NumberOfSimulations)
                                    DefaultScenarios_Pool = np.where(Defaults > LGD_Pool_min)[0]
                                    DefaultScenarios_Indiv = np.where(Defaults == 1)[0]
                                    LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                    # Direct simulation of LGDs for scenarios with only one default
                                    if len(DefaultScenarios_Indiv) > 0:
                                        lgd[DefaultScenarios_Indiv]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                                np.sqrt(group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios_Indiv, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i]) +
                                                np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios_Indiv]
                                            )) / np.sqrt(1 - group_data.rho.iloc[i])
                                        )
                                    # Simulation of LGDs if all scenarios are subject to aggregated treatment
                                    if (not LGD_Pool) or (np.min(Defaults)>LGD_Pool_min):
                                        # Calculation of conditional LGD
                                        lgd_cond[DefaultScenarios]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                        )
                                        # Calculation of conditional LGD variance
                                        ConditionalLGDvariance=1/Defaults[DefaultScenarios]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)]

                                        # Derivation of new rho parameters for aggregated LGD distribution 
                                        row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                        col_idx = np.ceil(2000*lgd_cond[DefaultScenarios]-0.5).astype(int)
                                        rho_new=RhoMapping[row_idx,col_idx]
                                    
                                        # Simulation of aggregated LGD
                                        lgd[DefaultScenarios]=fast_norm_cdf(
                                                (ndtri(lgd_cond[DefaultScenarios]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios] 
                                                ) / np.sqrt(1 - rho_new)
                                            )
                                    else:
                                        if len(DefaultScenarios_Pool) > 0:
                                            # Calculation of conditional LGD
                                            lgd_cond[DefaultScenarios_Pool]=fast_norm_cdf(
                                                (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios_Pool, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                            )
                                            # Calculation of conditional LGD variance
                                            ConditionalLGDvariance=1/Defaults[DefaultScenarios_Pool]*LGD_cond_var_Mapping[np.ceil(2000*group_data.rho.iloc[i]*(1-group_data.LGD_Rsquared.iloc[i])/(1-group_data.rho.iloc[i]*group_data.LGD_Rsquared.iloc[i])-0.5).astype(int),np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)]

                                            # Derivation of new rho parameters for aggregated LGD distribution 
                                            row_idx = np.ceil(2000*np.sqrt(ConditionalLGDvariance)-0.5).astype(int)
                                            col_idx = np.ceil(2000*lgd_cond[DefaultScenarios_Pool]-0.5).astype(int)
                                            rho_new=RhoMapping[row_idx,col_idx]
                                        
                                            # Simulation of aggregated LGD
                                            lgd[DefaultScenarios_Pool]=fast_norm_cdf(
                                                    (ndtri(lgd_cond[DefaultScenarios_Pool]) - np.sqrt(rho_new) * LGD_Return_idio[DefaultScenarios_Pool] 
                                                    ) / np.sqrt(1 - rho_new)
                                                )
                                        if np.min(DefaultScenarios)<=LGD_Pool_min and LGD_Pool_min>=2:
                                            for j in range(2,min(LGD_Pool_min,np.max(Defaults)+1)):
                                                DefaultScenarios_noPool = np.where(Defaults ==j)[0]
                                                if len(DefaultScenarios_noPool) > 0:
                                                    idiosyncratic = rng.normal(size=(len(DefaultScenarios_noPool), j))
                                                    systematic = (X.loc[DefaultScenarios_noPool, group_data.LGD_Segment.iloc[i]] - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                    latent = (np.sqrt(group_data.LGD_Rsquared.iloc[i]) * systematic.to_numpy()[:, np.newaxis] + np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * idiosyncratic)
                                                    numerator = ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * latent
                                                    cond_LGD_probs=fast_norm_cdf(numerator / np.sqrt(1 - group_data.rho.iloc[i]))
                                                    lgd[DefaultScenarios_noPool]=np.mean(cond_LGD_probs, axis=1)

                                    PoolLoss = Defaults*group_data.EAD.iloc[i]*lgd/group_data.Number_of_Exposures.iloc[i]

                                if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                    PoolLoss_default=PoolLoss.copy()
                                    PoolLoss+=PoolLoss_migration
                            
                                if not ImportanceSampling:
                                    ExpLoss_simulated[group_data.index[i], run-1] = np.sum(PoolLoss) / NumberOfSimulations
                                else:
                                    ExpLoss_simulated[group_data.index[i], run-1] = np.sum(PoolLoss * PortfolioLoss[:, NumberOfSeeds + run-1])

                                # Expected shortfall contribution
                                if EScontrib:
                                    if EScontrib_sim:
                                        for j in range(NumberOfAllocationQuantiles):
                                            scenario_idx = ES_scenarios[(run - 1) * NumberOfAllocationQuantiles + j]

                                            if not ImportanceSampling:
                                                EScontrib_allocation[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    np.sum(PoolLoss[scenario_idx]) / len(scenario_idx)
                                                    - PortfolioData.loc[group_data.index[i], 'EAD'] * PortfolioData.loc[group_data.index[i], 'LGD'] * PortfolioData.loc[group_data.index[i], 'PD']
                                                )
                                            else:
                                                weights = PortfolioLoss[scenario_idx, NumberOfSeeds + run-1]
                                                EScontrib_allocation[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    fast_weighted_avg(PoolLoss[scenario_idx],weights)
                                                    - PortfolioData.loc[group_data.index[i], 'EAD'] * PortfolioData.loc[group_data.index[i], 'LGD'] * PortfolioData.loc[group_data.index[i], 'PD']
                                                )
                                    if CondPDLGD:
                                        for j in range(NumberOfAllocationQuantiles):
                                            scenario_idx = ES_scenarios[(run - 1) * NumberOfAllocationQuantiles + j]

                                            if not ImportanceSampling:
                                                EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    np.sum(Defaults[scenario_idx]/group_data.Number_of_Exposures.iloc[i]) / len(scenario_idx)
                                                )
                                                if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = group_data.LGD.iloc[i]
                                                else: 
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                        fast_weighted_avg(lgd[scenario_idx],Defaults[scenario_idx])
                                                    ) if EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] > 0 else 0.0
                                            else:
                                                weights = PortfolioLoss[scenario_idx, NumberOfSeeds + run-1]
                                                EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    fast_weighted_avg(Defaults[scenario_idx],weights)/group_data.Number_of_Exposures.iloc[i]
                                                )
                                                if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = group_data.LGD.iloc[i]
                                                else: 
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                        fast_weighted_avg(lgd[scenario_idx], Defaults[scenario_idx] * weights) 
                                                    ) if EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] > 0 else 0.0
                                    
                                    if EScontrib_analytic:
                                        for j in range(NumberOfAllocationQuantiles):
                                            idx = j + (run - 1) * NumberOfAllocationQuantiles
                                            if Deterministic or group_data.LGD.iloc[i] in [0, 1]:
                                                help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= ES_boundary[run-1, j])[0]
                                                help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= ES_boundary[run-1, j] - group_data.EAD.iloc[i]*group_data.LGD.iloc[i]/group_data.Number_of_Exposures.iloc[i])[0]
                                                if not ImportanceSampling:
                                                    EScontrib_allocation_analytic[group_data.index[i], idx] = (
                                                        group_data.EAD.iloc[i]*group_data.LGD.iloc[i] *
                                                        np.sum(ConditionalPD[help_scenarios2]) /
                                                        (len(help_scenarios1) + np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1]))
                                                        - group_data.EAD.iloc[i]*group_data.LGD.iloc[i]*group_data.PD.iloc[i] 
                                                    ) 
                                                else:
                                                    num = fast_vector_mult(ConditionalPD[help_scenarios2],PortfolioLoss[help_scenarios2, NumberOfSeeds + run-1])
                                                    den = (
                                                        np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1]) +
                                                        num -
                                                        fast_vector_mult(ConditionalPD[help_scenarios1],PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1])
                                                    )
                                                    EScontrib_allocation_analytic[group_data.index[i], idx] = (
                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * num / den -
                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * group_data.PD.iloc[i] 
                                                    )
                                                if CondPDLGD:
                                                    if not ImportanceSampling:
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            np.sum(ConditionalPD[help_scenarios2]) /
                                                            (len(help_scenarios1) + np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1]))
                                                        )
                                                    else:
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            fast_vector_mult(ConditionalPD[help_scenarios2],PortfolioLoss[help_scenarios2, NumberOfSeeds + run-1]) /
                                                            (np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], PortfolioLoss[help_scenarios2, NumberOfSeeds + run-1]) -
                                                            fast_vector_mult(ConditionalPD[help_scenarios1], PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1]))
                                                        )
                                                    EScontrib_LGD_analytic[group_data.index[i], idx] = group_data.LGD.iloc[i]
                                            else:
                                                help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= ES_boundary[run-1, j])[0]
                                                help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= ES_boundary[run-1, j] - group_data.EAD.iloc[i]/group_data.Number_of_Exposures.iloc[i])[0]
                                                LGD = np.zeros(NumberOfSimulations)
                                                #Calculation of conditional LGD for scenarios that can exceed the ES_boundary
                                                if PLC:
                                                    X_diff = (X.loc[help_scenarios2, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                    LGD[help_scenarios2] = fast_norm_cdf(
                                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * X_diff
                                                        ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                    )
                                                else:
                                                    LGD[help_scenarios2] = fast_norm_cdf(
                                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y[help_scenarios2, group_data.LGD_Segment.iloc[i]].values
                                                        ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                    )
                                                #Scenarios in which client i has to default in order to push losses above ES_boundary
                                                help_scenarios2 = np.setdiff1d(help_scenarios2, help_scenarios1)
                                                CondProb_help = np.zeros(NumberOfSimulations)
                                                #Minimum simulated LGD that is required for the scenario to exceed ES_boundary
                                                z_term = (ES_boundary[run-1, j] - (PortfolioLoss[help_scenarios2, run-1] - PoolLoss[help_scenarios2]/group_data.Number_of_Exposures.iloc[i])) / (group_data.EAD.iloc[i]/group_data.Number_of_Exposures.iloc[i])
                                                z_term = np.clip(z_term, 0, 1)
                                                #Determine the parameter for the Vasicek distribution that describes the conditional LGD
                                                corr = group_data.rho.iloc[i] * (1 - group_data.LGD_Rsquared.iloc[i]) / (1 - group_data.rho.iloc[i]  * group_data.LGD_Rsquared.iloc[i])
                                                #Probability that the LGD exceeds the required level such that PortfolioLoss exceeds the ES_boundary
                                                CondProb_help[help_scenarios2] = fast_norm_cdf(
                                                    (ndtri(LGD[help_scenarios2]) - np.sqrt(1 - corr) * ndtri(z_term)) / np.sqrt(corr) 
                                                )

                                                if not ImportanceSampling:
                                                    numer = (fast_vector_mult(ConditionalPD[help_scenarios1], LGD[help_scenarios1]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2],
                                                                    Bivarcumnorm(ndtri(LGD[help_scenarios2]),
                                                                                ndtri(CondProb_help[help_scenarios2]),
                                                                                np.full(len(help_scenarios2), np.sqrt(corr))))) #Bivariate normal distribution yields the ES of a Vasicek distribution
                                                    denom = (len(help_scenarios1) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], CondProb_help[help_scenarios2]))
                                                else:
                                                    help_vec1=ConditionalPD[help_scenarios1]*LGD[help_scenarios1]
                                                    help_vec2=ConditionalPD[help_scenarios2] * PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]
                                                    numer = (fast_vector_mult(help_vec1, PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec2,
                                                                    Bivarcumnorm(ndtri(LGD[help_scenarios2]),
                                                                                ndtri(CondProb_help[help_scenarios2]),
                                                                                np.full(len(help_scenarios2), np.sqrt(corr)))))
                                                    denom = (np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec2, CondProb_help[help_scenarios2]))

                                                EScontrib_allocation_analytic[group_data.index[i], idx] = group_data.EAD.iloc[i] * numer / denom - \
                                                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * group_data.PD.iloc[i] if denom !=0 else 0.0

                                                if CondPDLGD:
                                                    if not ImportanceSampling:
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            (np.sum(ConditionalPD[help_scenarios1]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], CondProb_help[help_scenarios2])) /
                                                            (len(help_scenarios1) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], CondProb_help[help_scenarios2]))
                                                        )
                                                    else:
                                                        help_vec=ConditionalPD[help_scenarios2] * PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]
                                                        help_var=fast_vector_mult(help_vec,CondProb_help[help_scenarios2])
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            (fast_vector_mult(ConditionalPD[help_scenarios1], PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) + help_var) /
                                                            (np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) + help_var)
                                                        )

                                                    EScontrib_LGD_analytic[group_data.index[i], idx] = (
                                                        EScontrib_allocation_analytic[group_data.index[i], idx] +
                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * group_data.PD.iloc[i]
                                                    ) / (group_data.EAD.iloc[i] * EScontrib_PD_analytic[group_data.index[i], idx]) if EScontrib_PD_analytic[group_data.index[i], idx]>0 else 0.0                         
                                
                                if WCE:
                                    if WCE_sim:
                                        for j in range(NumberOfWCE_windows):
                                            window_index = j + (run - 1) * NumberOfWCE_windows
                                            scenarios = WCE_scenarios[window_index]
                                            
                                            if not ImportanceSampling:
                                                WCE_allocation[group_data.index[i], window_index] = np.sum(PoolLoss[scenarios]) / len(scenarios)
                                            else:
                                                weights = PortfolioLoss[scenarios, NumberOfSeeds + run-1]
                                                WCE_allocation[group_data.index[i], window_index] = fast_weighted_avg(PoolLoss[scenarios],weights) 

                                        if CondPDLGD:
                                            for j in range(NumberOfWCE_windows):
                                                window_index = j + (run - 1) * NumberOfWCE_windows
                                                scenarios = WCE_scenarios[window_index]
                                                
                                                if not ImportanceSampling:
                                                    WCE_PD[group_data.index[i], window_index] = np.sum(Defaults[scenarios])/(group_data.Number_of_Exposures.iloc[i]* len(scenarios)) if len(scenarios)>0 else 0.0
                                                    
                                                    if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                        WCE_LGD[group_data.index[i], window_index] = group_data.LGD.iloc[i]
                                                    else:
                                                        # Conditional LGD: ratio of sum(LGD) to sum(PD)
                                                        denominator = np.sum(Defaults[scenarios])
                                                        if denominator > 0:
                                                            WCE_LGD[group_data.index[i], window_index] = fast_vector_mult(lgd[scenarios],Defaults[scenarios]) / denominator
                                                        else:
                                                            WCE_LGD[group_data.index[i], window_index] = 0.0
                                                else:
                                                    weights = PortfolioLoss[scenarios, NumberOfSeeds + run - 1]
                                                    weighted_PD = fast_vector_mult(Defaults[scenarios], weights)/group_data.Number_of_Exposures.iloc[i]
                                                    
                                                    # Weighted conditional PD
                                                    WCE_PD[group_data.index[i], window_index] = weighted_PD / np.sum(weights) if np.sum(weights) > 0 else 0.0

                                                    if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                        WCE_LGD[group_data.index[i], window_index] = group_data.LGD.iloc[i]
                                                    else:
                                                        help_vec=lgd[scenarios] * Defaults[scenarios]/group_data.Number_of_Exposures.iloc[i]
                                                        weighted_LGD_numerator = fast_vector_mult(help_vec, weights)
                                                                                                            
                                                        # Weighted conditional LGD
                                                        if weighted_PD > 0:
                                                            WCE_LGD[group_data.index[i], window_index] = weighted_LGD_numerator / weighted_PD
                                                        else:
                                                            WCE_LGD[group_data.index[i], window_index] = 0.0 
                                        
                                    if WCE_analytic:
                                        for j in range(NumberOfWCE_windows):
                                            LGD = group_data.LGD.iloc[i]
                                            ead = group_data.EAD.iloc[i]
                                            ead_lgd = ead * LGD
                                            window_index = j + (run - 1) * NumberOfWCE_windows

                                            if Deterministic or LGD == 0 or LGD == 1:
                                                if WCE_upper[run-1, j] - WCE_lower[run-1, j] >= ead_lgd/group_data.Number_of_Exposures.iloc[i]:
                                                    help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] > WCE_upper[run-1, j])[0]
                                                    help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] > WCE_upper[run-1, j] - ead_lgd/group_data.Number_of_Exposures.iloc[i])[0]
                                                    help_scenarios3 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= WCE_lower[run-1, j])[0]
                                                    help_scenarios4 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= WCE_lower[run-1, j] - ead_lgd/group_data.Number_of_Exposures.iloc[i])[0]
                                                    if not ImportanceSampling:
                                                        numerator = ead_lgd * (np.sum(ConditionalPD[help_scenarios4]) - np.sum(ConditionalPD[help_scenarios2]))
                                                        denominator = (np.sum(1 - ConditionalPD[help_scenarios2]) - np.sum(1 - ConditionalPD[help_scenarios1])
                                                                    + len(help_scenarios3) - len(help_scenarios2)
                                                                    + np.sum(ConditionalPD[help_scenarios4]) - np.sum(ConditionalPD[help_scenarios3]))
                                                    else:
                                                        w = PortfolioLoss[:, NumberOfSeeds + run-1]
                                                        numerator = ead_lgd * (fast_vector_mult(ConditionalPD[help_scenarios4], w[help_scenarios4])
                                                                            - fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2]))
                                                        denominator = (fast_vector_mult(w[help_scenarios2], (1 - ConditionalPD[help_scenarios2]))
                                                                    - fast_vector_mult(w[help_scenarios1], (1 - ConditionalPD[help_scenarios1]))
                                                                    + np.sum(w[help_scenarios3]) - np.sum(w[help_scenarios2])
                                                                    + fast_vector_mult(ConditionalPD[help_scenarios4], w[help_scenarios4])
                                                                    - fast_vector_mult(ConditionalPD[help_scenarios3], w[help_scenarios3]))

                                                    WCE_allocation_analytic[group_data.index[i], window_index] = numerator / denominator if denominator != 0 else 0

                                                    if CondPDLGD:
                                                        WCE_PD_analytic[group_data.index[i], window_index] = WCE_allocation_analytic[group_data.index[i], window_index] / ead_lgd
                                                        WCE_LGD_analytic[group_data.index[i], window_index] = LGD

                                                else:
                                                    help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] > WCE_upper[run-1, j] - ead_lgd/group_data.Number_of_Exposures.iloc[i])[0]
                                                    help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= WCE_lower[run-1, j] - ead_lgd/group_data.Number_of_Exposures.iloc[i])[0]

                                                    scenarios = WCE_scenarios[(run-1) * NumberOfWCE_windows + j]

                                                    if not ImportanceSampling:
                                                        numerator = ead_lgd * (np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1]))
                                                        denominator = (np.sum(1 - ConditionalPD[scenarios])
                                                                    + np.sum(ConditionalPD[help_scenarios2])
                                                                    - np.sum(ConditionalPD[help_scenarios1]))
                                                    else:
                                                        w = PortfolioLoss[:, NumberOfSeeds + run-1]
                                                        numerator = ead_lgd * (fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2])
                                                                            - fast_vector_mult(ConditionalPD[help_scenarios1], w[help_scenarios1]))
                                                        denominator = (fast_vector_mult((1 - ConditionalPD[scenarios]), w[scenarios])
                                                                    + fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2])
                                                                    - fast_vector_mult(ConditionalPD[help_scenarios1], w[help_scenarios1]))

                                                    WCE_allocation_analytic[group_data.index[i], window_index] = numerator / denominator if denominator != 0 else 0

                                                    if CondPDLGD:
                                                        if not ImportanceSampling:
                                                            WCE_PD_analytic[group_data.index[i], window_index] = (np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1])) / denominator if denominator != 0 else 0
                                                        else:
                                                            WCE_PD_analytic[group_data.index[i], window_index] = (fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2])
                                                                                    - fast_vector_mult(ConditionalPD[help_scenarios1], w[help_scenarios1])) / denominator if denominator != 0 else 0
                                                        WCE_LGD_analytic[group_data.index[i], window_index] = LGD
                                            else:
                                                for j in range(NumberOfWCE_windows):
                                                    idx = group_data.index[i]

                                                    help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] > WCE_upper[run-1, j])[0]
                                                    help_scenarios2 = np.where(
                                                        PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] > max(WCE_lower[run-1, j], WCE_upper[run-1, j] - group_data.EAD.iloc[i]/group_data.Number_of_Exposures.iloc[i])
                                                    )[0]
                                                    help_scenarios3 = np.where(PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= WCE_lower[run-1, j])[0]
                                                    help_scenarios4 = np.where(
                                                        PortfolioLoss[:, run-1] - PoolLoss/group_data.Number_of_Exposures.iloc[i] >= WCE_lower[run-1, j] - group_data.EAD.iloc[i]/group_data.Number_of_Exposures.iloc[i]
                                                    )[0]
                                                    #Conditional LGD
                                                    LGD = np.zeros(NumberOfSimulations)
                                                    if PLC:
                                                        LGD[help_scenarios4] = fast_norm_cdf(
                                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * (X.loc[help_scenarios4, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                        )
                                                    else:
                                                        LGD[help_scenarios4] = fast_norm_cdf(
                                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y.loc[help_scenarios4, group_data.LGD_Segment.iloc[i]].values
                                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                        )
                                                    
                                                    CondProb_help_upper = np.zeros(NumberOfSimulations)
                                                    CondProb_help_lower = np.zeros(NumberOfSimulations)
                                                
                                                    #Vasicek parameter for conditional LGD distribution
                                                    alpha = np.sqrt((group_data.rho.iloc[i] * (1 - group_data.LGD_Rsquared.iloc[i])) /
                                                                (1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]))

                                                    upper_arg = np.clip(
                                                        (WCE_upper[run - 1, j] - (PortfolioLoss[help_scenarios4, run - 1] - PoolLoss[help_scenarios4]/group_data.Number_of_Exposures.iloc[i])) /
                                                        (group_data.EAD.iloc[i]/group_data.Number_of_Exposures.iloc[i]), 0, 1
                                                    )
                                                    lower_arg = np.clip(
                                                        (WCE_lower[run - 1, j] - (PortfolioLoss[help_scenarios4, run - 1] - PoolLoss[help_scenarios4]/group_data.Number_of_Exposures.iloc[i])) /
                                                        (group_data.EAD.iloc[i]/group_data.Number_of_Exposures.iloc[i]), 0, 1
                                                    )

                                                    CondProb_help_upper[help_scenarios4] = fast_norm_cdf((
                                                        ndtri(LGD[help_scenarios4]) - np.sqrt(1 - alpha**2) * ndtri(upper_arg)
                                                    ) / alpha)

                                                    CondProb_help_lower[help_scenarios4] = fast_norm_cdf((
                                                        ndtri(LGD[help_scenarios4]) - np.sqrt(1 - alpha**2) * ndtri(lower_arg)
                                                    ) / alpha)

                                                    help_scenarios4 = np.setdiff1d(help_scenarios4, help_scenarios3)
                                                    help_scenarios3 = np.setdiff1d(help_scenarios3, help_scenarios2)
                                                    help_scenarios2 = np.setdiff1d(help_scenarios2, help_scenarios1)

                                                    if not ImportanceSampling:
                                                        help_vec1=(LGD[help_scenarios2] - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios2]),
                                                                ndtri(CondProb_help_upper[help_scenarios2]),
                                                                np.full(len(help_scenarios2), alpha)
                                                            )
                                                        )
                                                        help_vec2=(
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_lower[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            ) - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_upper[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            )
                                                        )
                                                        help_vec3= CondProb_help_lower[help_scenarios4] - CondProb_help_upper[help_scenarios4]
                                                        numerator = (
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], help_vec1) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios3], LGD[help_scenarios3]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios4], help_vec2)
                                                        )
                                                        denominator = (
                                                            np.sum(1 - ConditionalPD[help_scenarios2]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], (1 - CondProb_help_upper[help_scenarios2])) +
                                                            len(help_scenarios3) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios4],help_vec3)
                                                        )
                                                    else:
                                                        help_vec0=ConditionalPD[help_scenarios2] * PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]
                                                        help_vec1=(LGD[help_scenarios2] - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios2]),
                                                                ndtri(CondProb_help_upper[help_scenarios2]),
                                                                np.full(len(help_scenarios2), alpha)
                                                            )
                                                        )
                                                        help_vec2=(
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_lower[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            ) - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_upper[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            )
                                                        )
                                                        help_vec3 = CondProb_help_lower[help_scenarios4] - CondProb_help_upper[help_scenarios4]
                                                        help_vec4 = ConditionalPD[help_scenarios3] * LGD[help_scenarios3]
                                                        help_vec5 = PortfolioLoss[help_scenarios4, NumberOfSeeds + run - 1] * ConditionalPD[help_scenarios4]
                                                        numerator = (
                                                            fast_vector_mult(help_vec0, help_vec1) +
                                                            fast_vector_mult(help_vec4, PortfolioLoss[help_scenarios3, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec5, help_vec2) 
                                                        )
                                                        denominator = (
                                                            fast_vector_mult((1 - ConditionalPD[help_scenarios2]), PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec0, (1 - CondProb_help_upper[help_scenarios2])) +
                                                            np.sum(PortfolioLoss[help_scenarios3, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec5, help_vec3)
                                                        )

                                                    WCE_allocation_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] = group_data.EAD.iloc[i] * numerator / denominator if denominator > 0 else 0.0

                                                    if CondPDLGD:
                                                        if not ImportanceSampling:
                                                            pd_numerator = (
                                                                fast_vector_mult(ConditionalPD[help_scenarios2], (1 - CondProb_help_upper[help_scenarios2])) +
                                                                np.sum(ConditionalPD[help_scenarios3]) +
                                                                fast_vector_mult(ConditionalPD[help_scenarios4],help_vec3)
                                                            )
                                                        else:
                                                            pd_numerator = (
                                                                fast_vector_mult(help_vec0, (1 - CondProb_help_upper[help_scenarios2])) +
                                                                fast_vector_mult(ConditionalPD[help_scenarios3], PortfolioLoss[help_scenarios3, NumberOfSeeds + run - 1]) +
                                                                fast_vector_mult(help_vec5, help_vec3)
                                                            )
                                                        
                                                        WCE_PD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] = pd_numerator / denominator if denominator > 0 else 0.0
                                                        WCE_LGD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] = (
                                                            WCE_allocation_analytic[group_data.index[i], j +(run-1) * NumberOfWCE_windows] /
                                                            (group_data.EAD.iloc[i] * WCE_PD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows])
                                                        ) if WCE_PD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] > 0 else 0.0

                                # Marginal risk contribution (simulated)
                                if MRcontribSim:
                                    index = (run - 1) * NumberOfAllocationQuantiles

                                    if not ImportanceSampling:
                                        cov_value = np.cov(PoolLoss, PortfolioLoss[:, run-1])[0, 1]
                                        MRcontribSim_allocation[group_data.index[i], index] = cov_value / StdDev_simulated[run-1, 0]
                                    else:
                                        help_vec = PoolLoss * PortfolioLoss[:, NumberOfSeeds + run-1]
                                        weighted_product = fast_vector_mult(PortfolioLoss[:, run-1], help_vec)
                                        weighted_expected = EL_simulated[run-1, 0] * fast_vector_mult(PoolLoss, PortfolioLoss[:, NumberOfSeeds + run-1])
                                        MRcontribSim_allocation[group_data.index[i], index] = (weighted_product - weighted_expected) / StdDev_simulated[run-1, 0]
                    
                        else: # Modelling for individual clients
                            M=group_data.shape[0]
                            Return_idio=rng.normal(size=NumberOfSimulations) # Simulation of idiosyncratic returns
                            for i in range(0,M):
                                if Progress_bar_counter>0:
                                    elapsed = time.time() - start_time_progress
                                    est_total_time = elapsed / Progress_bar_counter * PortfolioData.shape[0]
                                    eta_seconds = est_total_time - elapsed
                                    eta_str = format_eta(eta_seconds)
                                    Progress_bar.set(value=Progress_bar_counter,message=f"Capital allocation run {run}", detail=f"ETA {eta_str}")
                                Progress_bar_counter += 1
                                if group_data.PD.iloc[i]<1:
                                    Return_total=math.sqrt(group_data.Rsquared.iloc[i])*X[group_data.Segment.iloc[i]]+math.sqrt(1-group_data.Rsquared.iloc[i])*Return_idio
                                    Defaults=(Return_total<ndtri(group_data.PD.iloc[i])).astype(int)
                                    if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                        Migrations=Migration_events(Return_total, Migration_boundaries[int(group_data.RatingClass.iloc[i])-1,:], Defaults)
                                        PoolLoss_migration = MigrationEffects[group_data.index[i],Migrations-1]
                                else:
                                    Defaults = np.ones(NumberOfSimulations, dtype=int)
                                    if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                        PoolLoss_migration=np.zeros(NumberOfSimulations) #Defaulted clients cannot have migration effects
                                if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                    PoolLoss = Defaults*group_data.EAD.iloc[i]*group_data.LGD.iloc[i]
                                elif Vasicek:
                                    DefaultScenarios = np.where(Defaults > 0)[0]
                                    lgd=np.zeros(NumberOfSimulations)
                                    LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                    if len(DefaultScenarios) > 0:
                                        lgd[DefaultScenarios]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                                np.sqrt(group_data.LGD_Rsquared.iloc[i]) * Y.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values +
                                                np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios]
                                            )) / np.sqrt(1 - group_data.rho.iloc[i])
                                        )
                                    PoolLoss = Defaults*group_data.EAD.iloc[i]*lgd
                                elif PLC:
                                    DefaultScenarios = np.where(Defaults > 0)[0]
                                    lgd=np.zeros(NumberOfSimulations)
                                    LGD_Return_idio=rng.normal(size=NumberOfSimulations)
                                    if len(DefaultScenarios) > 0:
                                        lgd[DefaultScenarios]=fast_norm_cdf(
                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i]) * (
                                                np.sqrt(group_data.LGD_Rsquared.iloc[i]) * (X.loc[DefaultScenarios, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i]) +
                                                np.sqrt(1 - group_data.LGD_Rsquared.iloc[i]) * LGD_Return_idio[DefaultScenarios]
                                            )) / np.sqrt(1 - group_data.rho.iloc[i])
                                        )
                                    PoolLoss = Defaults*group_data.EAD.iloc[i]*lgd

                                PoolLoss=ensure_numpy(PoolLoss)
                                Defaults=ensure_numpy(Defaults)

                                if MarketValueApproach and group_data.Approach.iloc[i]=="Market value":
                                    PoolLoss_default=PoolLoss.copy()
                                    PoolLoss+=PoolLoss_migration
                                                            
                                if not ImportanceSampling:
                                    ExpLoss_simulated[group_data.index[i], run-1] = np.sum(PoolLoss) / NumberOfSimulations
                                else:
                                    ExpLoss_simulated[group_data.index[i], run-1] = np.sum(PoolLoss * PortfolioLoss[:, NumberOfSeeds + run-1])
                                
                                if EScontrib_analytic or WCE_analytic:
                                    if group_data.PD.iloc[i]<1:
                                        ConditionalPD = fast_norm_cdf(
                                            (ndtri(group_data.PD.iloc[i]) - np.sqrt(group_data.Rsquared.iloc[i]) * X[group_data.Segment.iloc[i]].values) / np.sqrt(1 - group_data.Rsquared.iloc[i])
                                        )
                                    else:
                                        ConditionalPD = np.ones(NumberOfSimulations)
                                
                                
                                # Expected shortfall contribution
                                if EScontrib:
                                    if EScontrib_sim:
                                        for j in range(NumberOfAllocationQuantiles):
                                            scenario_idx = ES_scenarios[(run - 1) * NumberOfAllocationQuantiles + j]

                                            if not ImportanceSampling:
                                                EScontrib_allocation[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    np.sum(PoolLoss[scenario_idx]) / len(scenario_idx)
                                                    - PortfolioData.loc[group_data.index[i], 'EAD'] * PortfolioData.loc[group_data.index[i], 'LGD'] * PortfolioData.loc[group_data.index[i], 'PD']
                                                )
                                            else:
                                                weights = PortfolioLoss[scenario_idx, NumberOfSeeds + run-1]
                                                EScontrib_allocation[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    fast_weighted_avg(PoolLoss[scenario_idx],weights)
                                                    - PortfolioData.loc[group_data.index[i], 'EAD'] * PortfolioData.loc[group_data.index[i], 'LGD'] * PortfolioData.loc[group_data.index[i], 'PD']
                                                )
                                    if CondPDLGD:
                                        for j in range(NumberOfAllocationQuantiles):
                                            scenario_idx = ES_scenarios[(run - 1) * NumberOfAllocationQuantiles + j]

                                            if not ImportanceSampling:
                                                EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    np.sum(Defaults[scenario_idx]) / len(scenario_idx)
                                                )
                                                if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = group_data.LGD.iloc[i]
                                                else: 
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                        fast_weighted_avg(lgd[scenario_idx],Defaults[scenario_idx])
                                                    ) if EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] > 0 else 0.0
                                            else:
                                                weights = PortfolioLoss[scenario_idx, NumberOfSeeds + run-1]
                                                EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                    fast_weighted_avg(Defaults[scenario_idx],weights)
                                                )
                                                if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = group_data.LGD.iloc[i]
                                                else: 
                                                    EScontrib_LGD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] = (
                                                        fast_weighted_avg(lgd[scenario_idx], Defaults[scenario_idx] * weights) 
                                                    ) if EScontrib_PD[group_data.index[i], j + (run - 1) * NumberOfAllocationQuantiles] > 0 else 0.0
                                    
                                    if EScontrib_analytic:
                                        for j in range(NumberOfAllocationQuantiles):
                                            idx = j + (run - 1) * NumberOfAllocationQuantiles
                                            if Deterministic or group_data.LGD.iloc[i] in [0, 1]:
                                                help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= ES_boundary[run-1, j])[0]
                                                help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= ES_boundary[run-1, j] - group_data.EAD.iloc[i]*group_data.LGD.iloc[i])[0]
                                                if not ImportanceSampling:
                                                    EScontrib_allocation_analytic[group_data.index[i], idx] = (
                                                        group_data.EAD.iloc[i]*group_data.LGD.iloc[i] *
                                                        np.sum(ConditionalPD[help_scenarios2]) /
                                                        (len(help_scenarios1) + np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1]))
                                                        - group_data.EAD.iloc[i]*group_data.LGD.iloc[i]*group_data.PD.iloc[i] 
                                                    ) 
                                                else:
                                                    num = fast_vector_mult(ConditionalPD[help_scenarios2],PortfolioLoss[help_scenarios2, NumberOfSeeds + run-1])
                                                    den = (
                                                        np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1]) +
                                                        num -
                                                        fast_vector_mult(ConditionalPD[help_scenarios1],PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1])
                                                    )
                                                    EScontrib_allocation_analytic[group_data.index[i], idx] = (
                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * num / den -
                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * group_data.PD.iloc[i] 
                                                    )
                                                if CondPDLGD:
                                                    if not ImportanceSampling:
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            np.sum(ConditionalPD[help_scenarios2]) /
                                                            (len(help_scenarios1) + np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1]))
                                                        )
                                                    else:
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            fast_vector_mult(ConditionalPD[help_scenarios2],PortfolioLoss[help_scenarios2, NumberOfSeeds + run-1]) /
                                                            (np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], PortfolioLoss[help_scenarios2, NumberOfSeeds + run-1]) -
                                                            fast_vector_mult(ConditionalPD[help_scenarios1], PortfolioLoss[help_scenarios1, NumberOfSeeds + run-1]))
                                                        )
                                                    EScontrib_LGD_analytic[group_data.index[i], idx] = group_data.LGD.iloc[i]
                                            else:
                                                help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= ES_boundary[run-1, j])[0]
                                                help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= ES_boundary[run-1, j] - group_data.EAD.iloc[i])[0]
                                                LGD = np.zeros(NumberOfSimulations)
                                                #Calculation of conditional LGD for scenarios that can exceed the ES_boundary
                                                if PLC:
                                                    X_diff = (X.loc[help_scenarios2, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                    LGD[help_scenarios2] = fast_norm_cdf(
                                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * X_diff
                                                        ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                    )
                                                else:
                                                    LGD[help_scenarios2] = fast_norm_cdf(
                                                        (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y[help_scenarios2, group_data.LGD_Segment.iloc[i]].values
                                                        ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                    )
                                                #Scenarios in which client i has to default in order to push losses above ES_boundary
                                                help_scenarios2 = np.setdiff1d(help_scenarios2, help_scenarios1)
                                                CondProb_help = np.zeros(NumberOfSimulations)
                                                #Minimum simulated LGD that is required for the scenario to exceed ES_boundary
                                                z_term = (ES_boundary[run-1, j] - (PortfolioLoss[help_scenarios2, run-1] - PoolLoss[help_scenarios2])) / group_data.EAD.iloc[i]
                                                z_term = np.clip(z_term, 0, 1)
                                                #Determine the parameter for the Vasicek distribution that describes the conditional LGD
                                                corr = group_data.rho.iloc[i] * (1 - group_data.LGD_Rsquared.iloc[i]) / (1 - group_data.rho.iloc[i]  * group_data.LGD_Rsquared.iloc[i])
                                                #Probability that the LGD exceeds the required level such that PortfolioLoss exceeds the ES_boundary
                                                CondProb_help[help_scenarios2] = fast_norm_cdf(
                                                    (ndtri(LGD[help_scenarios2]) - np.sqrt(1 - corr) * ndtri(z_term)) / np.sqrt(corr) 
                                                )

                                                if not ImportanceSampling:
                                                    numer = (fast_vector_mult(ConditionalPD[help_scenarios1], LGD[help_scenarios1]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2],
                                                                    Bivarcumnorm(ndtri(LGD[help_scenarios2]),
                                                                                ndtri(CondProb_help[help_scenarios2]),
                                                                                np.full(len(help_scenarios2), np.sqrt(corr))))) #Bivariate normal distribution yields the ES of a Vasicek distribution
                                                    denom = (len(help_scenarios1) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], CondProb_help[help_scenarios2]))
                                                else:
                                                    help_vec1=ConditionalPD[help_scenarios1]*LGD[help_scenarios1]
                                                    help_vec2=ConditionalPD[help_scenarios2] * PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]
                                                    numer = (fast_vector_mult(help_vec1, PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec2,
                                                                    Bivarcumnorm(ndtri(LGD[help_scenarios2]),
                                                                                ndtri(CondProb_help[help_scenarios2]),
                                                                                np.full(len(help_scenarios2), np.sqrt(corr)))))
                                                    denom = (np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec2, CondProb_help[help_scenarios2]))

                                                EScontrib_allocation_analytic[group_data.index[i], idx] = group_data.EAD.iloc[i] * numer / denom - \
                                                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * group_data.PD.iloc[i] if denom !=0 else 0.0

                                                if CondPDLGD:
                                                    if not ImportanceSampling:
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            (np.sum(ConditionalPD[help_scenarios1]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], CondProb_help[help_scenarios2])) /
                                                            (len(help_scenarios1) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], CondProb_help[help_scenarios2]))
                                                        )
                                                    else:
                                                        help_vec=ConditionalPD[help_scenarios2] * PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]
                                                        help_var=fast_vector_mult(help_vec,CondProb_help[help_scenarios2])
                                                        EScontrib_PD_analytic[group_data.index[i], idx] = (
                                                            (fast_vector_mult(ConditionalPD[help_scenarios1], PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) + help_var) /
                                                            (np.sum(PortfolioLoss[help_scenarios1, NumberOfSeeds + run - 1]) + help_var)
                                                        )

                                                    EScontrib_LGD_analytic[group_data.index[i], idx] = (
                                                        EScontrib_allocation_analytic[group_data.index[i], idx] +
                                                        group_data.EAD.iloc[i] * group_data.LGD.iloc[i] * group_data.PD.iloc[i]
                                                    ) / (group_data.EAD.iloc[i] * EScontrib_PD_analytic[group_data.index[i], idx]) if EScontrib_PD_analytic[group_data.index[i], idx]>0 else 0.0                         
                                
                                if WCE:
                                    if WCE_sim:
                                        for j in range(NumberOfWCE_windows):
                                            window_index = j + (run - 1) * NumberOfWCE_windows
                                            scenarios = WCE_scenarios[window_index]
                                            
                                            if not ImportanceSampling:
                                                WCE_allocation[group_data.index[i], window_index] = np.sum(PoolLoss[scenarios]) / len(scenarios)
                                            else:
                                                weights = PortfolioLoss[scenarios, NumberOfSeeds + run-1]
                                                WCE_allocation[group_data.index[i], window_index] = fast_weighted_avg(PoolLoss[scenarios],weights) 

                                        if CondPDLGD:
                                            for j in range(NumberOfWCE_windows):
                                                window_index = j + (run - 1) * NumberOfWCE_windows
                                                scenarios = WCE_scenarios[window_index]
                                                
                                                if not ImportanceSampling:
                                                    WCE_PD[group_data.index[i], window_index] = np.sum(Defaults[scenarios])/len(scenarios) if len(scenarios)>0 else 0.0
                                                    
                                                    if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                        WCE_LGD[group_data.index[i], window_index] = group_data.LGD.iloc[i]
                                                    else:
                                                        # Conditional LGD: ratio of sum(LGD) to sum(PD)
                                                        denominator = np.sum(Defaults[scenarios])
                                                        if denominator > 0:
                                                            WCE_LGD[group_data.index[i], window_index] = fast_vector_mult(lgd[scenarios],Defaults[scenarios]) / denominator
                                                        else:
                                                            WCE_LGD[group_data.index[i], window_index] = 0.0
                                                else:
                                                    weights = PortfolioLoss[scenarios, NumberOfSeeds + run - 1]
                                                    weighted_PD = fast_vector_mult(Defaults[scenarios], weights)
                                                    
                                                    # Weighted conditional PD
                                                    WCE_PD[group_data.index[i], window_index] = weighted_PD / np.sum(weights) if np.sum(weights) > 0 else 0.0

                                                    if Deterministic or group_data.LGD.iloc[i]==0 or group_data.LGD.iloc[i]==1:
                                                        WCE_LGD[group_data.index[i], window_index] = group_data.LGD.iloc[i]
                                                    else:
                                                        help_vec=lgd[scenarios] * Defaults[scenarios]
                                                        weighted_LGD_numerator = fast_vector_mult(help_vec, weights)
                                                                                                            
                                                        # Weighted conditional LGD
                                                        if weighted_PD > 0:
                                                            WCE_LGD[group_data.index[i], window_index] = weighted_LGD_numerator / weighted_PD
                                                        else:
                                                            WCE_LGD[group_data.index[i], window_index] = 0.0 
                                        
                                    if WCE_analytic:
                                        for j in range(NumberOfWCE_windows):
                                            LGD = group_data.LGD.iloc[i]
                                            ead = group_data.EAD.iloc[i]
                                            ead_lgd = ead * LGD
                                            window_index = j + (run - 1) * NumberOfWCE_windows

                                            if Deterministic or LGD == 0 or LGD == 1:
                                                if WCE_upper[run-1, j] - WCE_lower[run-1, j] >= ead_lgd:
                                                    help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss > WCE_upper[run-1, j])[0]
                                                    help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss > WCE_upper[run-1, j] - ead_lgd)[0]
                                                    help_scenarios3 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= WCE_lower[run-1, j])[0]
                                                    help_scenarios4 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= WCE_lower[run-1, j] - ead_lgd)[0]
                                                    if not ImportanceSampling:
                                                        numerator = ead_lgd * (np.sum(ConditionalPD[help_scenarios4]) - np.sum(ConditionalPD[help_scenarios2]))
                                                        denominator = (np.sum(1 - ConditionalPD[help_scenarios2]) - np.sum(1 - ConditionalPD[help_scenarios1])
                                                                    + len(help_scenarios3) - len(help_scenarios2)
                                                                    + np.sum(ConditionalPD[help_scenarios4]) - np.sum(ConditionalPD[help_scenarios3]))
                                                    else:
                                                        w = PortfolioLoss[:, NumberOfSeeds + run-1]
                                                        numerator = ead_lgd * (fast_vector_mult(ConditionalPD[help_scenarios4], w[help_scenarios4])
                                                                            - fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2]))
                                                        denominator = (fast_vector_mult(w[help_scenarios2], (1 - ConditionalPD[help_scenarios2]))
                                                                    - fast_vector_mult(w[help_scenarios1], (1 - ConditionalPD[help_scenarios1]))
                                                                    + np.sum(w[help_scenarios3]) - np.sum(w[help_scenarios2])
                                                                    + fast_vector_mult(ConditionalPD[help_scenarios4], w[help_scenarios4])
                                                                    - fast_vector_mult(ConditionalPD[help_scenarios3], w[help_scenarios3]))

                                                    WCE_allocation_analytic[group_data.index[i], window_index] = numerator / denominator if denominator != 0 else 0

                                                    if CondPDLGD:
                                                        WCE_PD_analytic[group_data.index[i], window_index] = WCE_allocation_analytic[group_data.index[i], window_index] / ead_lgd
                                                        WCE_LGD_analytic[group_data.index[i], window_index] = LGD

                                                else:
                                                    help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss > WCE_upper[run-1, j] - ead_lgd)[0]
                                                    help_scenarios2 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= WCE_lower[run-1, j] - ead_lgd)[0]

                                                    scenarios = WCE_scenarios[(run-1) * NumberOfWCE_windows + j]

                                                    if not ImportanceSampling:
                                                        numerator = ead_lgd * (np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1]))
                                                        denominator = (np.sum(1 - ConditionalPD[scenarios])
                                                                    + np.sum(ConditionalPD[help_scenarios2])
                                                                    - np.sum(ConditionalPD[help_scenarios1]))
                                                    else:
                                                        w = PortfolioLoss[:, NumberOfSeeds + run-1]
                                                        numerator = ead_lgd * (fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2])
                                                                            - fast_vector_mult(ConditionalPD[help_scenarios1], w[help_scenarios1]))
                                                        denominator = (fast_vector_mult((1 - ConditionalPD[scenarios]), w[scenarios])
                                                                    + fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2])
                                                                    - fast_vector_mult(ConditionalPD[help_scenarios1], w[help_scenarios1]))

                                                    WCE_allocation_analytic[group_data.index[i], window_index] = numerator / denominator if denominator != 0 else 0

                                                    if CondPDLGD:
                                                        if not ImportanceSampling:
                                                            WCE_PD_analytic[group_data.index[i], window_index] = (np.sum(ConditionalPD[help_scenarios2]) - np.sum(ConditionalPD[help_scenarios1])) / denominator if denominator != 0 else 0
                                                        else:
                                                            WCE_PD_analytic[group_data.index[i], window_index] = (fast_vector_mult(ConditionalPD[help_scenarios2], w[help_scenarios2])
                                                                                    - fast_vector_mult(ConditionalPD[help_scenarios1], w[help_scenarios1])) / denominator if denominator != 0 else 0
                                                        WCE_LGD_analytic[group_data.index[i], window_index] = LGD
                                            else:
                                                for j in range(NumberOfWCE_windows):
                                                    idx = group_data.index[i]

                                                    help_scenarios1 = np.where(PortfolioLoss[:, run-1] - PoolLoss > WCE_upper[run-1, j])[0]
                                                    help_scenarios2 = np.where(
                                                        PortfolioLoss[:, run-1] - PoolLoss > max(WCE_lower[run-1, j], WCE_upper[run-1, j] - group_data.EAD.iloc[i])
                                                    )[0]
                                                    help_scenarios3 = np.where(PortfolioLoss[:, run-1] - PoolLoss >= WCE_lower[run-1, j])[0]
                                                    help_scenarios4 = np.where(
                                                        PortfolioLoss[:, run-1] - PoolLoss >= WCE_lower[run-1, j] - group_data.EAD.iloc[i]
                                                    )[0]
                                                    #Conditional LGD
                                                    LGD = np.zeros(NumberOfSimulations)
                                                    if PLC:
                                                        LGD[help_scenarios4] = fast_norm_cdf(
                                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * (X.loc[help_scenarios4, group_data.LGD_Segment.iloc[i]].values - group_data.mu_PLC.iloc[i]) / np.sqrt(group_data.sigma_PLC.iloc[i])
                                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                        )
                                                    else:
                                                        LGD[help_scenarios4] = fast_norm_cdf(
                                                            (ndtri(group_data.LGD.iloc[i]) - np.sqrt(group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]) * Y.loc[help_scenarios4, group_data.LGD_Segment.iloc[i]].values
                                                            ) / np.sqrt(1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i])
                                                        )
                                                    
                                                    CondProb_help_upper = np.zeros(NumberOfSimulations)
                                                    CondProb_help_lower = np.zeros(NumberOfSimulations)
                                                
                                                    #Vasicek parameter for conditional LGD distribution
                                                    alpha = np.sqrt((group_data.rho.iloc[i] * (1 - group_data.LGD_Rsquared.iloc[i])) /
                                                                (1 - group_data.rho.iloc[i] * group_data.LGD_Rsquared.iloc[i]))

                                                    upper_arg = np.clip(
                                                        (WCE_upper[run - 1, j] - (PortfolioLoss[help_scenarios4, run - 1] - PoolLoss[help_scenarios4])) /
                                                        (group_data.EAD.iloc[i]), 0, 1
                                                    )
                                                    lower_arg = np.clip(
                                                        (WCE_lower[run - 1, j] - (PortfolioLoss[help_scenarios4, run - 1] - PoolLoss[help_scenarios4])) /
                                                        (group_data.EAD.iloc[i]), 0, 1
                                                    )

                                                    CondProb_help_upper[help_scenarios4] = fast_norm_cdf((
                                                        ndtri(LGD[help_scenarios4]) - np.sqrt(1 - alpha**2) * ndtri(upper_arg)
                                                    ) / alpha)

                                                    CondProb_help_lower[help_scenarios4] = fast_norm_cdf((
                                                        ndtri(LGD[help_scenarios4]) - np.sqrt(1 - alpha**2) * ndtri(lower_arg)
                                                    ) / alpha)

                                                    help_scenarios4 = np.setdiff1d(help_scenarios4, help_scenarios3)
                                                    help_scenarios3 = np.setdiff1d(help_scenarios3, help_scenarios2)
                                                    help_scenarios2 = np.setdiff1d(help_scenarios2, help_scenarios1)

                                                    if not ImportanceSampling:
                                                        help_vec1=(LGD[help_scenarios2] - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios2]),
                                                                ndtri(CondProb_help_upper[help_scenarios2]),
                                                                np.full(len(help_scenarios2), alpha)
                                                            )
                                                        )
                                                        help_vec2=(
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_lower[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            ) - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_upper[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            )
                                                        )
                                                        help_vec3= CondProb_help_lower[help_scenarios4] - CondProb_help_upper[help_scenarios4]
                                                        numerator = (
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], help_vec1) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios3], LGD[help_scenarios3]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios4], help_vec2)
                                                        )
                                                        denominator = (
                                                            np.sum(1 - ConditionalPD[help_scenarios2]) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios2], (1 - CondProb_help_upper[help_scenarios2])) +
                                                            len(help_scenarios3) +
                                                            fast_vector_mult(ConditionalPD[help_scenarios4],help_vec3)
                                                        )
                                                    else:
                                                        help_vec0=ConditionalPD[help_scenarios2] * PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]
                                                        help_vec1=(LGD[help_scenarios2] - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios2]),
                                                                ndtri(CondProb_help_upper[help_scenarios2]),
                                                                np.full(len(help_scenarios2), alpha)
                                                            )
                                                        )
                                                        help_vec2=(
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_lower[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            ) - 
                                                            Bivarcumnorm(
                                                                ndtri(LGD[help_scenarios4]),
                                                                ndtri(CondProb_help_upper[help_scenarios4]),
                                                                np.full(len(help_scenarios4), alpha)
                                                            )
                                                        )
                                                        help_vec3 = CondProb_help_lower[help_scenarios4] - CondProb_help_upper[help_scenarios4]
                                                        help_vec4 = ConditionalPD[help_scenarios3] * LGD[help_scenarios3]
                                                        help_vec5 = PortfolioLoss[help_scenarios4, NumberOfSeeds + run - 1] * ConditionalPD[help_scenarios4]
                                                        numerator = (
                                                            fast_vector_mult(help_vec0, help_vec1) +
                                                            fast_vector_mult(help_vec4, PortfolioLoss[help_scenarios3, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec5, help_vec2) 
                                                        )
                                                        denominator = (
                                                            fast_vector_mult((1 - ConditionalPD[help_scenarios2]), PortfolioLoss[help_scenarios2, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec0, (1 - CondProb_help_upper[help_scenarios2])) +
                                                            np.sum(PortfolioLoss[help_scenarios3, NumberOfSeeds + run - 1]) +
                                                            fast_vector_mult(help_vec5, help_vec3)
                                                        )

                                                    WCE_allocation_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] = group_data.EAD.iloc[i] * numerator / denominator if denominator > 0 else 0.0

                                                    if CondPDLGD:
                                                        if not ImportanceSampling:
                                                            pd_numerator = (
                                                                fast_vector_mult(ConditionalPD[help_scenarios2], (1 - CondProb_help_upper[help_scenarios2])) +
                                                                np.sum(ConditionalPD[help_scenarios3]) +
                                                                fast_vector_mult(ConditionalPD[help_scenarios4],help_vec3)
                                                            )
                                                        else:
                                                            pd_numerator = (
                                                                fast_vector_mult(help_vec0, (1 - CondProb_help_upper[help_scenarios2])) +
                                                                fast_vector_mult(ConditionalPD[help_scenarios3], PortfolioLoss[help_scenarios3, NumberOfSeeds + run - 1]) +
                                                                fast_vector_mult(help_vec5, help_vec3)
                                                            )
                                                        
                                                        WCE_PD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] = pd_numerator / denominator if denominator > 0 else 0.0
                                                        WCE_LGD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] = (
                                                            WCE_allocation_analytic[group_data.index[i], j +(run-1) * NumberOfWCE_windows] /
                                                            (group_data.EAD.iloc[i] * WCE_PD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows])
                                                        ) if WCE_PD_analytic[group_data.index[i], j + (run-1) * NumberOfWCE_windows] > 0 else 0.0

                                # Marginal risk contribution (simulated)
                                if MRcontribSim:
                                    index = (run - 1) * NumberOfAllocationQuantiles

                                    if not ImportanceSampling:
                                        cov_value = np.cov(PoolLoss, PortfolioLoss[:, run-1])[0, 1]
                                        MRcontribSim_allocation[group_data.index[i], index] = cov_value / StdDev_simulated[run-1, 0]
                                    else:
                                        help_vec = PoolLoss * PortfolioLoss[:, NumberOfSeeds + run-1]
                                        weighted_product = fast_vector_mult(PortfolioLoss[:, run-1], help_vec)
                                        weighted_expected = EL_simulated[run-1, 0] * fast_vector_mult(PoolLoss, PortfolioLoss[:, NumberOfSeeds + run-1])
                                        MRcontribSim_allocation[group_data.index[i], index] = (weighted_product - weighted_expected) / StdDev_simulated[run-1, 0]
                                            

            EndTime_CA=datetime.now()  
            RunTime = EndTime_CA - StartTime_CA

            hours, remainder = divmod(RunTime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            ui.notification_show(
                f"Capital allocation runs finished after {int(hours)}h:{int(minutes)}m:{int(round(seconds))}s",
                type="message",
                duration=5,
            )
            
           
            # Re-scaling of allocation to balance back to ECAP
            for run in range(1, NumberOfSeeds + 1):
                if EScontrib:
                    if EScontrib_sim:
                        for j in range(NumberOfAllocationQuantiles):
                            col_idx = j + (run-1) * NumberOfAllocationQuantiles
                            total = np.sum(EScontrib_allocation[:, col_idx])
                            if total != 0:
                                EScontrib_allocation[:, col_idx] *= ECAP[run-1, j] / total

                    if EScontrib_analytic:
                        for j in range(NumberOfAllocationQuantiles):
                            col_idx = j + (run-1) * NumberOfAllocationQuantiles
                            total = np.sum(EScontrib_allocation_analytic[:, col_idx])
                            if total != 0:
                                EScontrib_allocation_analytic[:, col_idx] *= ECAP[run-1, j] / total

                if MRcontribSim:
                    MRcontribSim_help = MRcontribSim_allocation[:, (run-1) * NumberOfAllocationQuantiles]
                    for j in range(NumberOfAllocationQuantiles):
                        col_idx = j + (run-1) * NumberOfAllocationQuantiles
                        total = np.sum(MRcontribSim_help)
                        if total != 0:
                            MRcontribSim_allocation[:, col_idx] = MRcontribSim_help * ECAP[run-1, j] / total                                                                

            # Average allocation across all seeds
            ExpLoss_simulated[:, NumberOfSeeds] = np.sum(ExpLoss_simulated[:, :NumberOfSeeds], axis=1) / NumberOfSeeds

            if EScontrib:
                if EScontrib_sim:
                    for j in range(NumberOfAllocationQuantiles):
                        cols = [j + q * NumberOfAllocationQuantiles for q in range(NumberOfSeeds)]
                        avg_col = j + NumberOfSeeds * NumberOfAllocationQuantiles
                        EScontrib_allocation[:, avg_col] = np.sum(EScontrib_allocation[:, cols], axis=1) / NumberOfSeeds
                        if CondPDLGD:
                            EScontrib_PD[:, avg_col] = np.sum(EScontrib_PD[:, cols], axis=1) / NumberOfSeeds
                            EScontrib_LGD[:, avg_col] = np.sum(EScontrib_LGD[:, cols], axis=1) / NumberOfSeeds

                if EScontrib_analytic:
                    for j in range(NumberOfAllocationQuantiles):
                        cols = [j + q * NumberOfAllocationQuantiles for q in range(NumberOfSeeds)]
                        avg_col = j + NumberOfSeeds * NumberOfAllocationQuantiles
                        EScontrib_allocation_analytic[:, avg_col] = np.sum(EScontrib_allocation_analytic[:, cols], axis=1) / NumberOfSeeds
                        if CondPDLGD:
                            EScontrib_PD_analytic[:, avg_col] = np.sum(EScontrib_PD_analytic[:, cols], axis=1) / NumberOfSeeds
                            EScontrib_LGD_analytic[:, avg_col] = np.sum(EScontrib_LGD_analytic[:, cols], axis=1) / NumberOfSeeds

            if WCE:
                if WCE_sim:
                    for j in range(NumberOfWCE_windows):
                        cols = [j + q * NumberOfWCE_windows for q in range(NumberOfSeeds)]
                        avg_col = j + NumberOfSeeds * NumberOfWCE_windows
                        WCE_allocation[:, avg_col] = np.sum(WCE_allocation[:, cols], axis=1) / NumberOfSeeds
                        if CondPDLGD:
                            WCE_PD[:, avg_col] = np.sum(WCE_PD[:, cols], axis=1) / NumberOfSeeds
                            WCE_LGD[:, avg_col] = np.sum(WCE_LGD[:, cols], axis=1) / NumberOfSeeds

                if WCE_analytic:
                    for j in range(NumberOfWCE_windows):
                        cols = [j + q * NumberOfWCE_windows for q in range(NumberOfSeeds)]
                        avg_col = j + NumberOfSeeds * NumberOfWCE_windows
                        WCE_allocation_analytic[:, avg_col] = np.sum(WCE_allocation_analytic[:, cols], axis=1) / NumberOfSeeds
                        if CondPDLGD:
                            WCE_PD_analytic[:, avg_col] = np.sum(WCE_PD_analytic[:, cols], axis=1) / NumberOfSeeds
                            WCE_LGD_analytic[:, avg_col] = np.sum(WCE_LGD_analytic[:, cols], axis=1) / NumberOfSeeds

            if MRcontribSim:
                for j in range(NumberOfAllocationQuantiles):
                    cols = [j + q * NumberOfAllocationQuantiles for q in range(NumberOfSeeds)]
                    avg_col = j + NumberOfSeeds * NumberOfAllocationQuantiles
                    MRcontribSim_allocation[:, avg_col] = np.sum(MRcontribSim_allocation[:, cols], axis=1) / NumberOfSeeds

            # Collection of allocation results and preparation of output file
            ClusterResults = PortfolioData.copy()

            # Add analytical and simulated expected loss
            el_columns = ["Analytical expected loss"]
            el_columns += [f"Simulated expected loss (run {run})" for run in range(1, NumberOfSeeds + 1)]
            el_columns += ["Simulated expected loss (average)"]

            ClusterResults[el_columns] = np.column_stack([EL_analytical, ExpLoss_simulated])

            # Simulated marginal risk contributions
            if MRcontribSim:
                mr_columns = []
                for run in range(1, NumberOfSeeds + 1):
                    mr_columns += [f"Simulated marginal risk contribution ({(q * 100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                mr_columns += [f"Simulated marginal risk contribution ({(q * 100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                ClusterResults[mr_columns] = MRcontribSim_allocation

            # Expected shortfall contributions
            if EScontrib:
                if EScontrib_sim:
                    es_sim_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        es_sim_columns += [f"Expected shortfall contribution ({(q * 100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    es_sim_columns += [f"Expected shortfall contribution ({(q * 100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    ClusterResults[es_sim_columns] = EScontrib_allocation

                if EScontrib_analytic:
                    es_an_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        es_an_columns += [f"Expected shortfall contribution, deterministic approximation ({(q * 100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    es_an_columns += [f"Expected shortfall contribution, deterministic approximation ({(q * 100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    ClusterResults[es_an_columns] = EScontrib_allocation_analytic

            # WCE contributions
            if WCE:
                if WCE_sim:
                    wce_sim_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        wce_sim_columns += [f"WCE loss ({(l*100)}%-{(u*100)}%, run {run})" for l, u in zip(LowerBoundary[:NumberOfWCE_windows], UpperBoundary[:NumberOfWCE_windows])]
                    wce_sim_columns += [f"WCE loss ({(l*100)}%-{(u*100)}%, average)" for l, u in zip(LowerBoundary[:NumberOfWCE_windows], UpperBoundary[:NumberOfWCE_windows])]
                    ClusterResults[wce_sim_columns] = WCE_allocation

                if WCE_analytic:
                    wce_an_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        wce_an_columns += [f"WCE loss, deterministic approximation ({(l*100)}%-{(u*100)}%, run {run})" for l, u in zip(LowerBoundary[:NumberOfWCE_windows], UpperBoundary[:NumberOfWCE_windows])]
                    wce_an_columns += [f"WCE loss, deterministic approximation ({(l*100)}%-{(u*100)}%, average)" for l, u in zip(LowerBoundary[:NumberOfWCE_windows], UpperBoundary[:NumberOfWCE_windows])]
                    ClusterResults[wce_an_columns] = WCE_allocation_analytic

            if EScontrib:
                if EScontrib_sim and CondPDLGD:
                    # Conditional default rates from ES simulation
                    pd_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        pd_columns += [f"Conditional default rates (ES, {(q*100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    pd_columns += [f"Conditional default rates (ES, {(q*100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    ClusterResults[pd_columns] = EScontrib_PD

                    # Conditional loss rates from ES simulation
                    lgd_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        lgd_columns += [f"Conditional loss rates (ES, {(q*100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    lgd_columns += [f"Conditional loss rates (ES, {(q*100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    ClusterResults[lgd_columns] = EScontrib_LGD

                if EScontrib_analytic and CondPDLGD:
                    # Conditional default rates from analytic ES
                    pd_an_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        pd_an_columns += [f"Conditional default rates (ES, deterministic approximation, {(q*100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    pd_an_columns += [f"Conditional default rates (ES, deterministic approximation, {(q*100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    ClusterResults[pd_an_columns] = EScontrib_PD_analytic

                    # Conditional loss rates from analytic ES
                    lgd_an_columns = []
                    for run in range(1, NumberOfSeeds + 1):
                        lgd_an_columns += [f"Conditional loss rates (ES, deterministic approximation, {(q*100)}%, run {run})" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    lgd_an_columns += [f"Conditional loss rates (ES, deterministic approximation, {(q*100)}%, average)" for q in Quantiles[:NumberOfAllocationQuantiles]]
                    ClusterResults[lgd_an_columns] = EScontrib_LGD_analytic

            if WCE and WCE_sim:
                if CondPDLGD:
                    # Conditional default rates (WCE, run-specific)
                    wce_pd_columns = [
                        f"Conditional default rates (WCE, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, run {run})"
                        for i in range(NumberOfWCE_windows)
                        for run in range(1, NumberOfSeeds + 1)
                    ]
                    wce_pd_columns += [
                        f"Conditional default rates (WCE, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, average)"
                        for i in range(NumberOfWCE_windows)
                    ]
                    ClusterResults[wce_pd_columns] = WCE_PD

                    # Conditional loss rates (WCE, run-specific)
                    wce_lgd_columns = [
                        f"Conditional loss rates (WCE, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, run {run})"
                        for i in range(NumberOfWCE_windows)
                        for run in range(1, NumberOfSeeds + 1)
                    ]
                    wce_lgd_columns += [
                        f"Conditional loss rates (WCE, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, average)"
                        for i in range(NumberOfWCE_windows)
                    ]
                    ClusterResults[wce_lgd_columns] = WCE_LGD

            if WCE and WCE_analytic:
                if CondPDLGD:
                    # Conditional default rates (WCE, deterministic approximation, run-specific)
                    wce_pd_an_columns = [
                        f"Conditional default rates (WCE, deterministic approximation, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, run {run})"
                        for i in range(NumberOfWCE_windows)
                        for run in range(1, NumberOfSeeds + 1)
                    ]
                    wce_pd_an_columns += [
                        f"Conditional default rates (WCE, deterministic approximation, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, average)"
                        for i in range(NumberOfWCE_windows)
                    ]
                    ClusterResults[wce_pd_an_columns] = WCE_PD_analytic

                    # Conditional loss rates (WCE, deterministic approximation, run-specific)
                    wce_lgd_an_columns = [
                        f"Conditional loss rates (WCE, deterministic approximation, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, run {run})"
                        for i in range(NumberOfWCE_windows)
                        for run in range(1, NumberOfSeeds + 1)
                    ]
                    wce_lgd_an_columns += [
                        f"Conditional loss rates (WCE, deterministic approximation, {(LowerBoundary[i] * 100)}%-{(UpperBoundary[i] * 100)}%, average)"
                        for i in range(NumberOfWCE_windows)
                    ]
                    ClusterResults[wce_lgd_an_columns] = WCE_LGD_analytic

            timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
            file_name_ClusterResults = f"Allocation_Results_{timestamp}.csv"
            
            @render.download(filename=file_name_ClusterResults)
            def downloadAllocationResults():
                yield ClusterResults.to_csv() 

            #ClusterResults.to_csv(file_name_ClusterResults, index=False)


            # Create summary table for allocation results
            @render.ui
            def Allocation_summary():
                df = ClusterResults.copy()
                
                def row_to_html(row, bold=False):
                    style = "font-weight:bold;" if bold else ""
                    cells = "".join(
                        f"<td style='text-align:right; {style}'>{value:,.2f}</td>"
                        if isinstance(value, (int, float)) and not math.isnan(value)
                        else f"<td style='{style}'>{'' if pd.isna(value) else value}</td>"
                        for value in row
                    )
                    return f"<tr>{cells}</tr>"

                # Build HTML table manually
                headers = "".join(
                    f"<th style='text-align:left'>{col}</th>" if i != 0 else f"<th>{col}</th>"
                    for i, col in enumerate(df.columns)
                )
                rows = [
                    row_to_html(df.iloc[i], bold=False)
                    for i in range(len(df))
                ]

                table_html = f"""
                <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                    <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                        <thead style="position: sticky; top: 0; background-color: white; z-index: 1;">
                            <tr>{headers}</tr>
                        </thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
                """
                return ui.HTML(table_html)

            # Create summary table for ES allocation results
            if EScontrib==True:

                row_names = [f"Run {i+1}" for i in range(NumberOfSeeds)] 
                column_names = (               
                    [f"{(q*100)}%" for q in Quantiles[:NumberOfAllocationQuantiles]]
                )

                EScontrib_Interval = pd.DataFrame(EScontrib_Interval, index=row_names, columns=column_names)

                @render.ui
                def EScontrib_summary():
                    df = EScontrib_Interval.copy()
                    df.index.name = "Run"
                    df = df.reset_index()

                    def row_to_html(row, bold=False):
                        style = "font-weight:bold;" if bold else ""
                        cells = "".join(
                            f"<td style='text-align:center; {style}'>{int(value):,}</td>"
                            if isinstance(value, (int, float))
                            else f"<td style='{style}'>{value}</td>"
                            for value in row
                        )
                        return f"<tr>{cells}</tr>"

                    # Build HTML table manually
                    headers = "".join(
                        f"<th style='text-align:center'>{col}</th>" if i != 0 else f"<th>{col}</th>"
                        for i, col in enumerate(df.columns)
                    )
                    rows = [
                        row_to_html(df.iloc[i], bold=False)
                        for i in range(len(df))
                    ]

                    table_html = f"""
                    <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                        <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                            <thead><tr>{headers}</tr></thead>
                            <tbody>{"".join(rows)}</tbody>
                        </table>
                    </div>
                    """
                    return ui.HTML(table_html)
                
            # Create summary table for WCE allocation results
            if WCE==True:
                row_names = [f"Run {i+1}" for i in range(NumberOfSeeds)] 
                column_names = (
                    [f"{(q1*100)}% - {(q2*100)}%" for q1, q2 in zip(LowerBoundary[:NumberOfWCE_windows], UpperBoundary[:NumberOfWCE_windows])]
                )

                WCE_Interval = pd.DataFrame(WCE_Interval, index=row_names, columns=column_names)

                @render.ui
                def WCE_summary():
                    df = WCE_Interval.copy()
                    df.index.name = "Run"
                    df = df.reset_index()

                    def row_to_html(row, bold=False):
                        style = "font-weight:bold;" if bold else ""
                        cells = "".join(
                            f"<td style='text-align:center; {style}'>{int(value):,}</td>"
                            if isinstance(value, (int, float))
                            else f"<td style='{style}'>{value}</td>"
                            for value in row
                        )
                        return f"<tr>{cells}</tr>"

                    # Build HTML table manually
                    headers = "".join(
                        f"<th style='text-align:center'>{col}</th>" if i != 0 else f"<th>{col}</th>"
                        for i, col in enumerate(df.columns)
                    )
                    rows = [
                        row_to_html(df.iloc[i], bold=False)
                        for i in range(len(df))
                    ]

                    table_html = f"""
                    <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                        <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                            <thead><tr>{headers}</tr></thead>
                            <tbody>{"".join(rows)}</tbody>
                        </table>
                    </div>
                    """
                    return ui.HTML(table_html)
                
            # Create table for allocation stability analysis
            if NumberOfSeeds > 1:
                total_cols = 1 + (
                    MRcontribSim +
                    EScontrib * (EScontrib_sim + EScontrib_analytic)
                ) * NumberOfAllocationQuantiles + (
                    WCE * (WCE_sim + WCE_analytic) * NumberOfWCE_windows
                )
                AllocationStability = np.zeros((5, total_cols))
                ColumnName = ["EL"]

                if Deterministic:
                    Stability_IDs = ClusterResults[ClusterResults["PD"] < 1].index
                else:
                    pd1 = ClusterResults["PD"] == 1
                    lgd_in_range = (ClusterResults["LGD"] < 1) & (ClusterResults["LGD"] > 0)
                    Stability_IDs = ClusterResults[
                        (ClusterResults["PD"] < 1) | (pd1 & lgd_in_range)
                    ].index

                # Helper to extract columns
                def make_columns(prefix, run_ids):
                    return [f"{prefix}run {i})" for i in run_ids]

                # Compute CoV helper
                def get_cov_vector(columns):
                    values = ClusterResults[columns]
                    cov = values.std(axis=1) / values.mean(axis=1).abs()
                    return cov.fillna(0)
                
                # EL CoV
                el_columns = make_columns("Simulated expected loss (", range(1, NumberOfSeeds + 1))
                cov_vector = get_cov_vector(el_columns)
                AllocationStability[:, 0] = [
                    cov_vector.loc[Stability_IDs].mean(),
                    cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] == 1)].mean(),
                    cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] > 1)].mean(),
                    cov_vector.loc[Stability_IDs].max(),
                    cov_vector.loc[Stability_IDs].min()
                ]
                col_idx = 1 

                # Marginal risk contributions
                if MRcontribSim:
                    for q in Quantiles[:NumberOfAllocationQuantiles]:
                        ColumnName.append(f"MRC ({q * 100}%)")
                        cols = make_columns(f"Simulated marginal risk contribution ({q * 100}%, ", range(1, NumberOfSeeds + 1))
                        cov_vector = get_cov_vector(cols)
                        AllocationStability[:, col_idx] = [
                            cov_vector.loc[Stability_IDs].mean(),
                            cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] == 1)].mean(),
                            cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] > 1)].mean(),
                            cov_vector.loc[Stability_IDs].max(),
                            cov_vector.loc[Stability_IDs].min()
                        ]
                        col_idx += 1
                    
                # Expected shortfall contributions (simulated)
                if EScontrib and EScontrib_sim:
                    for q in Quantiles[:NumberOfAllocationQuantiles]:
                        ColumnName.append(f"ESC ({q * 100}%)")
                        cols = make_columns(f"Expected shortfall contribution ({q * 100}%, ", range(1, NumberOfSeeds + 1))
                        cov_vector = get_cov_vector(cols)
                        AllocationStability[:, col_idx] = [
                            cov_vector.loc[Stability_IDs].mean(),
                            cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] == 1)].mean(),
                            cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] > 1)].mean(),
                            cov_vector.loc[Stability_IDs].max(),
                            cov_vector.loc[Stability_IDs].min()
                        ]
                        col_idx += 1

                # Expected shortfall contributions (analytic)
                if EScontrib and EScontrib_analytic:
                    for q in Quantiles[:NumberOfAllocationQuantiles]:
                        ColumnName.append(f"ESC_approx ({q * 100}%)")
                        cols = make_columns(f"Expected shortfall contribution, deterministic approximation ({q * 100}%, ", range(1, NumberOfSeeds + 1))
                        cov_vector = get_cov_vector(cols)
                        AllocationStability[:, col_idx] = [
                            cov_vector.loc[Stability_IDs].mean(),
                            cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] == 1)].mean(),
                            cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] > 1)].mean(),
                            cov_vector.loc[Stability_IDs].max(),
                            cov_vector.loc[Stability_IDs].min()
                        ]
                        col_idx += 1

                # WCE
                if WCE:
                    for i in range(NumberOfWCE_windows):
                        window_label = f"{LowerBoundary[i] * 100}%-{UpperBoundary[i] * 100}%"
                        if WCE_sim:
                            ColumnName.append(f"WCE ({window_label})")
                            cols = make_columns(f"WCE loss ({window_label}, ", range(1, NumberOfSeeds + 1))
                            cov_vector = get_cov_vector(cols)
                            AllocationStability[:, col_idx] = [
                                cov_vector.loc[Stability_IDs].mean(),
                                cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] == 1)].mean(),
                                cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] > 1)].mean(),
                                cov_vector.loc[Stability_IDs].max(),
                                cov_vector.loc[Stability_IDs].min()
                            ]
                            col_idx += 1

                        if WCE_analytic:
                            ColumnName.append(f"WCE_approx ({window_label})")
                            cols = make_columns(f"WCE loss, deterministic approximation ({window_label}, ", range(1, NumberOfSeeds + 1))
                            cov_vector = get_cov_vector(cols)
                            AllocationStability[:, col_idx] = [
                                cov_vector.loc[Stability_IDs].mean(),
                                cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] == 1)].mean(),
                                cov_vector.loc[ClusterResults.index.isin(Stability_IDs) & (ClusterResults["Number_of_Exposures"] > 1)].mean(),
                                cov_vector.loc[Stability_IDs].max(),
                                cov_vector.loc[Stability_IDs].min()
                            ]
                            col_idx += 1

                row_names = ["Mean (all)", "Mean (single names)", "Mean (pools)", "Max", "Min"]
                AllocationStability = pd.DataFrame(100*AllocationStability, columns=ColumnName, index=row_names)
                                
                @render.ui
                def Stability_summary():
                    df = AllocationStability.copy()
                    df.index.name = "Dimension"
                    df = df.reset_index()

                    def row_to_html(row, bold=False):
                        style = "font-weight:bold;" if bold else ""
                        cells = "".join(
                            f"<td style='text-align:center; {style}'>{value:,.1f}</td>"
                            if isinstance(value, (int, float)) and not math.isnan(value)
                            else f"<td style='{style}'>{'' if pd.isna(value) else value}</td>"
                            for value in row
                        )
                        return f"<tr>{cells}</tr>"

                    # Build HTML table manually
                    headers = "".join(
                        f"<th style='text-align:center'>{col}</th>" if i != 0 else f"<th>{col}</th>"
                        for i, col in enumerate(df.columns)
                    )
                    rows = [
                        row_to_html(df.iloc[i], bold=False)
                        for i in range(len(df))
                    ]

                    table_html = f"""
                    <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                        <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                            <thead><tr>{headers}</tr></thead>
                            <tbody>{"".join(rows)}</tbody>
                        </table>
                    </div>
                    """
                    return ui.HTML(table_html)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
                file_name_AllocationStability = f"Stability_summary_{timestamp}.csv"
                
                @render.download(filename=file_name_AllocationStability)
                def downloadAllocationStability():
                    yield AllocationStability.to_csv() 

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()


        output = s.getvalue()

        # Write selected lines (the main table) to CSV
        lines = output.splitlines()
        table_lines = [
            line for line in lines 
            if line.strip() and line.strip()[0].isdigit()  # crude filter for the data rows
        ]
        """ 
        with open("profile_output.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["ncalls", "tottime", "percall_1", "cumtime", "percall_2", "location"])
            
            # Write rows
            for line in table_lines:
                parts = line.split(None, 5)  # split into max 6 columns
                if len(parts) == 6:
                    writer.writerow(parts) """
        
        #print(s.getvalue())

        # Summary of run settings
        num_rows = 22
        Settings=pd.DataFrame("", columns=["Item", "Selected setting"],index=range(num_rows))
        Settings['Item']=[
            "Run type:",
            "Portfolio input:",
            "Correlation matrix:",
            "LGD correlation matrix:",
            "Migration matrix:",
            "Number of simulations:",
            "Number of seeds:",
            "Maximum pool size for individual treatment:",
            "Random seed:",
            "Credit migrations enabled:",
            "Importance sampling approach:",
            "Antithetic sampling:",
            "LGD settings:",
            "Expected shortfall contribution settings:",
            "Parallel computing settings:",
            "Pricing run settings:",
            "Pricing details:",
            "Pricing input:",
            "Start time:",
            "End time of capital window run:",
            "End time of total run:",
            "Total run time:"]
        
        if RunType1:
            Settings.loc[0,'Selected setting']="Capital window"
        elif RunType2:
            Settings.loc[0,'Selected setting']="Allocation"
        elif RunType3:
            Settings.loc[0,'Selected setting']="Stand-alone pricing"

        if not RunType3:
            if input.file1() is not None and len(input.file1()) > 0:
                Settings.loc[1,'Selected setting']=f"{input.file1()[0]['name']}"
            if input.file2() is not None and len(input.file2()) > 0:
                Settings.loc[2,'Selected setting']=f"{input.file2()[0]['name']}"
            if Vasicek:
                if input.file3() is not None and len(input.file3()) > 0:
                    Settings.loc[3,'Selected setting']=f"{input.file3()[0]['name']}"
            if MarketValueApproach:
                if input.file4() is not None and len(input.file4()) > 0:
                    Settings.loc[4,'Selected setting']=f"{input.file4()[0]['name']}"

        
        Settings.loc[5,'Selected setting']=f"{NumberOfSimulations:,}"
        Settings.loc[6,'Selected setting']=f"{NumberOfSeeds:,}"
        Settings.loc[7,'Selected setting']=f"{MaxPoolSize:,}"
        Settings.loc[8,'Selected setting']=f"{Rand_seed}"
        Settings.loc[9,'Selected setting']=f"{MarketValueApproach}"

        if not ImportanceSampling:
            Settings.loc[10,'Selected setting']="No importance sampling applied"
        elif ISType1:
            Settings.loc[10,'Selected setting']=f"Volatility scaling (Morokoff) with scaling factor = {ScalingFactor}"
        elif ISType2:
            Settings.loc[10,'Selected setting']=f"Mean shift (Kalkbrener, all factors) with amplification factor = {AmplificationFactor}"
        elif ISType3:
            Settings.loc[10,'Selected setting']=f"Mean shift (Kalkbrener, only first factor) with amplification factor = {AmplificationFactor}"
        
        Settings.loc[11,'Selected setting']=f"{AntitheticSampling}"

        if Deterministic:
            Settings.loc[12,'Selected setting']="Deterministic LGDs"
        elif Vasicek:
            if LGD_Pool:
                Settings.loc[12,'Selected setting']=f"Stochastic LGDs w/o PD/LGD correlations (pooled treatment for > {LGD_Pool_min:,} defaults)"
            else:
                Settings.loc[12,'Selected setting']="Stochastic LGDs w/o PD/LGD correlations (pooled treatment)"
        elif PLC:
            if LGD_Pool:
                Settings.loc[12,'Selected setting']=f"Stochastic LGDs with PD/LGD correlations (pooled treatment for > {LGD_Pool_min:,} defaults)"
            else:
                Settings.loc[12,'Selected setting']="Stochastic LGDs with PD/LGD correlations (pooled treatment)"

        if RunType1:
            Settings.loc[13,'Selected setting']="No capital allocations carried out"
        elif EScontrib:
            if ESLowerBoundary:
                Settings.loc[13,'Selected setting']=f"Expected Shortfall contribution (lower boundary = {LowerBoundary_ES:,})"
            else:
                if ESType1:
                    Settings.loc[13,'Selected setting']="Expected Shortfall contribution (VaR/ES match)"
                else:
                    Settings.loc[13,'Selected setting']="Expected Shortfall contribution (VaR allocation)"
        elif not EScontrib:
            Settings.loc[13,'Selected setting']="No Expected Shortfall allocations carried out"

        
        if not ParallelComp:
            Settings.loc[14,'Selected setting']="No parallel computing applied"
        else:
            if ParallelType1:
                Settings.loc[14,'Selected setting']=f"Parallel computation by seed using {NumberOfCores} cores"
            elif ParallelType3:
                Settings.loc[14,'Selected setting']=f"Parallel computation by seed and simulation step ({BatchSize_Sim1} batches) using {NumberOfCores} cores"
            elif ParallelType4:
                Settings.loc[14,'Selected setting']=f"Parallel computation by simulation step ({BatchSize_Sim} batches) using {NumberOfCores} cores"

        if PricingType0:
            Settings.loc[15,'Selected setting']="No pricing"
        elif PricingType1:
            Settings.loc[15,'Selected setting']="Pricing of new deal (full model run)"
        elif PricingType2:
            Settings.loc[15,'Selected setting']="Pricing of new deal (stand-alone analysis)"
        
        if not RunType1:
            if not PricingType0:
                Settings.loc[16,'Selected setting']=f"ExposureID={ExposureID_Pricing} / GroupID={GroupID_Pricing} / EAD={EAD_Pricing,:} / LGD={LGD_Pricing} / PD={PD_Pricing} / Segment={Segment_Pricing} / R-squared={Rsquared_Pricing} / LGD segment={LGD_Segment_Pricing} / k Parameter={k_Parameter_Pricing} / LGD R-squared={LGD_Rsquared_Pricing}/ Rating Class={RatingClass_Pricing} / Time-to-maturity={TimeToMaturity_Pricing} / Effective interest rate={Yield_Pricing} / Approach={Approach_Pricing}" 
        
        if PricingType2:
            if input.file6() is not None and len(input.file6()) > 0:
                Settings.loc[17,'Selected setting']=f"{input.file6()[0]['name']}"
            
        EndDate=datetime.now()
        Settings.loc[18,'Selected setting']=f"{StartDate.strftime("%Y-%m-%d %H:%M:%S")}"
        Settings.loc[19,'Selected setting']=f"{EndTime_CW.strftime("%Y-%m-%d %H:%M:%S")}"
        Settings.loc[20,'Selected setting']=f"{EndDate.strftime("%Y-%m-%d %H:%M:%S")}"

        duration = EndDate - StartDate
        total_seconds = int(duration.total_seconds())

        # Break down into hours, minutes, seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        Settings.loc[21,'Selected setting']=f"{hours}h:{minutes}m:{seconds}s"

        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        ui.notification_show(
            f"Model run completed after {int(hours)}h:{int(minutes)}m:{int(round(seconds))}s",
            type="message",
            duration=5,
        )
            


        @render.ui
        def Settings_summary():
            df = Settings.copy()
            
            def row_to_html(row, bold=False):
                style = "font-weight:bold;" if bold else ""
                cells = "".join(
                    f"<td style='text-align:center; {style}'>{value:,.1f}</td>"
                    if isinstance(value, (int, float)) and not math.isnan(value)
                    else f"<td style='{style}'>{'' if pd.isna(value) else value}</td>"
                    for value in row
                )
                return f"<tr>{cells}</tr>"

            # Build HTML table manually
            headers = "".join(
                f"<th style='text-align:left'>{col}</th>" if i != 0 else f"<th>{col}</th>"
                for i, col in enumerate(df.columns)
            )
            rows = [
                row_to_html(df.iloc[i], bold=False)
                for i in range(len(df))
            ]

            table_html = f"""
            <div style="max-height: 400px; overflow-y: auto; overflow-x: auto; white-space: nowrap;">
                <table class='table table-striped table-bordered' style="width:100%; table-layout: auto;">
                    <thead style="position: sticky; top: 0; background-color: white; z-index: 1;">
                        <tr>{headers}</tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            """
            return ui.HTML(table_html)

        timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
        file_name_Settings = f"Settings_{timestamp}.csv"
        
        @render.download(filename=file_name_Settings)
        def downloadSettings():
            yield Settings.to_csv() 

        # Combine all download files in zip file

        timestamp = datetime.now().strftime("%Y-%m-%d %H %M %S")
        file_name = f"Run_Results_{timestamp}.zip"

        @render.download(filename=file_name)
        def downloadData():
            buffer = io.BytesIO()

            if GraphicalOutput:
                pdf_bytes = generate_pdf_from_base64_images(images)

            with ZipFile(buffer, "w") as z:
                z.writestr(file_name_Settings, Settings.to_csv())

                z.writestr(file_name_PortfolioResults, PortfolioResults.to_csv())

                if LossDistribution:
                    z.writestr(file_name_LossOutput, LossOutput.to_csv()) 
                
                if LossDistribution2:
                    z.writestr(file_name_LossOutput2, LossOutput2.to_csv())
                
                if RunType2 or RunType3:
                    z.writestr(file_name_ClusterResults, ClusterResults.to_csv())
                    if NumberOfSeeds > 1:
                        z.writestr(file_name_AllocationStability, AllocationStability.to_csv())

                if GraphicalOutput:
                    z.writestr(filename_LossDistributions(), pdf_bytes)
                              

            buffer.seek(0)
            yield buffer.read()


          

    @output
    @render.ui
    def histograms():
        images = histogram_images.get()
        if not images:
            return ui.HTML("")
        img_html = "".join(f'<img src="data:image/png;base64,{img}" style="width: 100%; margin-bottom: 20px;" />' for img in images)
        return ui.HTML(img_html)
    






    
    
  

app = App(app_ui, server, static_assets=www_dir)

if __name__ == "__main__":
    from shiny import run_app
    run_app(app, port=8001, host="127.0.0.1")
