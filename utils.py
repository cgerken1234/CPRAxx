
###########################################
### Addtitional function for CPRAxx app ###
###########################################

import numpy as np
from scipy.stats import norm
from fpdf import FPDF
from PIL import Image
import io
import base64
import tempfile
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import ndtri
from numba import njit
from math import erf, sqrt

# Determination of quantile for optimal shift function
def quantile_search(alpha, PD, RSQ, tolerance, N):
    i = 2
    alpha_fit = 0.5
    difference = np.abs(
        np.sum(fast_norm_cdf((np.sqrt(RSQ) * ndtri(alpha_fit + np.arange(1, N + 1) * (0.99999 - alpha_fit) / N) + ndtri(PD)) / np.sqrt(1 - RSQ))) / N
        - fast_norm_cdf((np.sqrt(RSQ) * ndtri(alpha) + ndtri(PD)) / np.sqrt(1 - RSQ))
    )
    
    while difference > tolerance:
        if (np.sum(fast_norm_cdf((np.sqrt(RSQ) * ndtri(alpha_fit + np.arange(1, N + 1) * (0.99999 - alpha_fit) / N) + ndtri(PD)) / np.sqrt(1 - RSQ))) / N
            >= fast_norm_cdf((np.sqrt(RSQ) * ndtri(alpha) + ndtri(PD)) / np.sqrt(1 - RSQ))):
            alpha_fit -= 0.5 ** i
        else:
            alpha_fit += 0.5 ** i
        
        difference = np.abs(
            np.sum(fast_norm_cdf((np.sqrt(RSQ) * ndtri(alpha_fit + np.arange(1, N + 1) * (0.99999 - alpha_fit) / N) + ndtri(PD)) / np.sqrt(1 - RSQ))) / N
            - fast_norm_cdf((np.sqrt(RSQ) * ndtri(alpha) + ndtri(PD)) / np.sqrt(1 - RSQ))
        )
        i += 1
    
    return alpha_fit


# Determination of integral in order to derive optimal mean shift for systematic factors 
def optimal_shift(mu, alpha, PD, RSQ, N):
    N_hat = int(np.floor((1 - alpha) * N))
    intervals = np.arange(1, min(N_hat, N-1)+1) / N

    term1 = fast_norm_cdf((ndtri(PD) - np.sqrt(RSQ) * ndtri(intervals)) / np.sqrt(1 - RSQ)) ** 2
    term2 = np.exp(-0.5 * ndtri(intervals) ** 2)
    term3 = np.exp(-0.5 * (ndtri(intervals) - mu) ** 2)

    result = np.sum((term1 * term2) / term3) / len(intervals)
    return result

# Conversion of images into pdf
def generate_pdf_from_base64_images(base64_images):
    pdf = FPDF(unit="mm", format="A4")

    for img_str in base64_images:
        img_data = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_file:
            img.save(tmp_img_file, format="JPEG", dpi=(600, 600))
            tmp_img_path = tmp_img_file.name

        pdf.add_page()
        pdf.image(tmp_img_path, x=10, y=10, w=190)

    # Return PDF as byte string
    return pdf.output(dest="S").encode("latin1")



# Function to plot loss distribution based on variable inputs

def plot_loss_distribution(
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
):
    Unit_axis = 1
    Unit_axis_text = ""

    if run<NumberOfSeeds:
        PortfolioLoss_run=PortfolioLoss[:,run].copy()
        if VaR.iloc[run].max() > 1e6:
            Unit_axis = 1000
            Unit_axis_text = "(in thousands)"
        if VaR.iloc[run].max() > 1e9:
            Unit_axis = 1e6
            Unit_axis_text = "(in millions)"
        max_var = VaR.iloc[run].max()
    else: # To plot the total results
        PortfolioLoss_run=PortfolioLoss[:,range(NumberOfSeeds)].copy().flatten()
        if VaR.values.max() > 1e6:
            Unit_axis = 1000
            Unit_axis_text = "(in thousands)"
        if VaR.values.max() > 1e9:
            Unit_axis = 1e6
            Unit_axis_text = "(in millions)"
        max_var = VaR.values.max()

    min_loss = min(0, np.min(PortfolioLoss_run))
    
    if MarketValueApproach:
        Unit = np.ceil((max_var - min_loss) / 490 / Unit_axis) * Unit_axis
        bin_start = np.floor(min_loss / Unit)
        bins = np.concatenate([np.arange(bin_start, bin_start + 501) * Unit, [np.max(PortfolioLoss_run)]])
    else:
        Unit = np.ceil(max_var / 490 / Unit_axis) * Unit_axis
        bins = np.concatenate([np.arange(501) * Unit, [np.max(PortfolioLoss_run)]])

    if ImportanceSampling:
        bin_edges=bins
        counts=np.zeros(len(bin_edges)-1)
        if run<NumberOfSeeds:
            rows = np.where(PortfolioLoss[:,run] < bin_edges[1])[0]
            counts[0]=np.sum(PortfolioLoss[rows, NumberOfSeeds + run])
            for i in range(1,len(bin_edges)-2):
                rows = np.where(PortfolioLoss[:,run] < bin_edges[i])[0]
                counts[i]=np.sum(PortfolioLoss[rows, NumberOfSeeds + run])-np.sum(counts[range(i)])
            counts[len(bin_edges)-2]=np.sum(PortfolioLoss[:, NumberOfSeeds + run])-np.sum(counts[range(len(bin_edges)-2)])
        else:
            for run2 in range(NumberOfSeeds):
                rows = np.where(PortfolioLoss[:,run2] < bin_edges[1])[0]
                counts[0]+=np.sum(PortfolioLoss[rows, NumberOfSeeds + run2])
            for i in range(1,len(bin_edges)-2):
                for run2 in range(NumberOfSeeds):
                    rows = np.where(PortfolioLoss[:,run2] < bin_edges[i])[0]
                    counts[i]+=np.sum(PortfolioLoss[rows, NumberOfSeeds + run2])
                counts[i]=counts[i]-np.sum(counts[range(i)])
            counts[len(bin_edges)-2]=np.sum(PortfolioLoss[:, range(NumberOfSeeds,NumberOfSeeds + run)])-np.sum(counts[range(len(bin_edges)-2)])
        counts=counts*PortfolioLoss.shape[0] 
    else:
        counts, bin_edges = np.histogram(PortfolioLoss_run, bins=bins)
       
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=600)

    ax.plot(bin_centers[:500] / Unit_axis, counts[:500], color='darkblue', linewidth=2)
    
    if run<NumberOfSeeds:
        ax.set_xlim(min_loss / Unit_axis, ES[run, np.argmax(Quantiles)] / Unit_axis)
        ax.set_title(f"Loss distribution (run {run + 1})")
        x_ticks = np.ceil(ES[run, np.argmax(Quantiles)] / (4 * 250 * Unit_axis)) * 250 * np.arange(5)
    else:
        ax.set_xlim(min_loss / Unit_axis, np.average(ES[:, np.argmax(Quantiles)]) / Unit_axis)
        ax.set_title("Loss distribution (average)")
        x_ticks = np.ceil(np.average(ES[:, np.argmax(Quantiles)]) / (4 * 250 * Unit_axis)) * 250 * np.arange(5)    
    ax.set_xlabel(f"Loss amount {Unit_axis_text}")
    ax.set_ylabel("Number of simulation steps")
    
    y_max = np.max(counts[:500])
    y_ticks = np.ceil(y_max / 250) * 50 * np.arange(6)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ylims = ax.get_ylim()
    y_length = (ylims[1] - ylims[0])  

    # Expected Loss and Std Dev
    if run<NumberOfSeeds:
        el = EL_simulated[run]
        std = StdDev_simulated[run]
    else:
        el = np.average(EL_simulated)
        std = np.average(StdDev_simulated)

    ax.axvline(el / Unit_axis, color="firebrick", linewidth=1.5,ymax= (y_max-ylims[0])/y_length)
    ax.axvline((el + std) / Unit_axis, color="firebrick", linestyle="--",ymax=(y_max/3-ylims[0])/y_length)
  
    ax.text(el / Unit_axis+np.max(x_ticks)*0.01, 0.97 * y_max, f"Expected loss\n({int(round(float(el / Unit_axis))):,})",
            color="firebrick", ha="left", fontsize=9)

    ax.annotate(
        '',  # no text here
        xy=((el + std) / Unit_axis, y_max / 3),
        xytext=(el / Unit_axis, y_max / 3),
        arrowprops=dict(arrowstyle='->', color='firebrick', linestyle="--")
    )

    # Add the label above the arrow
    ax.text(
        (el + 0.5 * std) / Unit_axis,  # center above the arrow
        y_max / 3 + y_max * 0.02,      # a bit higher than the arrow
        f"Std.dev.\n({int(round(float(std / Unit_axis))):,})",
        ha='center',
        va='bottom',
        fontsize=9,
        color='firebrick'
    )

    # Plot VaR/ES/ECAP
    colors = ["orange", "cornflowerblue", "grey", "firebrick", "darkgreen"]
    for q in range(min(5, len(Quantiles))):
        quantiles_sorted = np.array(sorted(Quantiles, reverse=False))
        r = np.where(quantiles_sorted == Quantiles[q])[0][0]
        q_color = colors[r]
        q_label = f"{(Quantiles[q] * 100)}%"
        if run<NumberOfSeeds:
            var_q = VaR.iloc[run, q]
            es_q = ES[run, q]
            ecap_q = ECAP[run, q]
        else:
            el = np.average(EL_simulated)
            std = np.average(StdDev_simulated)
            var_q = np.average(VaR.iloc[:, q])
            es_q = np.average(ES[:, q])
            ecap_q = np.average(ECAP[:, q])

        ax.axvline(var_q / Unit_axis, color=q_color, linestyle="-", ymax=((0.7+r*0.05)*y_max-ylims[0])/y_length)
        ax.axvline(es_q / Unit_axis, color=q_color, linestyle="-", ymax=((0.4+r*0.05)*y_max-ylims[0])/y_length)

        ax.text(
            var_q / Unit_axis,  # center above the arrow
            (0.7+r*0.05)*y_max + y_max * 0.01,      # a bit higher than the arrow
            f"VaR ({q_label})",
            ha='center',
            va='bottom',
            fontsize=9,
            color=q_color
        )
        
        ax.text(
            es_q / Unit_axis,  # center above the arrow
            (0.4+r*0.05)*y_max + y_max * 0.01,      # a bit higher than the arrow
            f"ES ({q_label})",
            ha='center',
            va='bottom',
            fontsize=9,
            color=q_color
        )
        
        ax.annotate(
            '',  # no text here
            xy=((el + ecap_q) / Unit_axis, (0.7+r*0.05)*y_max),
            xytext=(el / Unit_axis, (0.7+r*0.05)*y_max),
            arrowprops=dict(arrowstyle='->', color=q_color)
        )

        ax.text(
            (el + 0.5 * ecap_q) / Unit_axis,  # center above the arrow
            (0.7+r*0.05)*y_max + y_max * 0.01,      # a bit higher than the arrow
           f"ECAP ({q_label}): {int(round(float(ecap_q / Unit_axis))):,}",
            ha='center',
            va='bottom',
            fontsize=9,
            color=q_color
        )

    return ax

# Function for fitting of asset correlations (used to derive rho parameter for Vasicek distribution)

def AC_fit(PD1, PD2, PDjoint, tolerance=1e-12, max_iter=100):
    # Check if the probabilities are valid
    if not (0 < PD1 < 1) or not (0 < PD2 < 1):
        raise ValueError("PD1 and PD2 must be between 0 and 1.")
    
    difference = 1
    i = 2
    # Initialize rho based on the condition
    if PD1 * PD2 <= PDjoint:
        rho = 0.5
    else:
        rho = -0.5
    
    iter_count = 0  # Counter for iterations
    
    while difference > tolerance and iter_count < max_iter:
        # Calculate the joint probability using bivariate normal CDF
        rho_matrix = np.array([[1, rho], [rho, 1]])
        joint_prob = multivariate_normal.cdf([ndtri(PD1), ndtri(PD2)], mean=[0, 0], cov=rho_matrix)
        
        # Update rho based on whether the computed joint probability is greater than or less than PDjoint
        if joint_prob >= PDjoint:
            rho -= 0.5 ** i
        else:
            rho += 0.5 ** i
        
        # Calculate the difference
        difference = abs(joint_prob - PDjoint)
        i += 1
        iter_count += 1
    
    if iter_count >= max_iter:
        print("Warning: Maximum number of iterations reached before convergence.")
    
    return rho

# Approximation for CDF of bivariate normal distribution (only valid for 0 < r < 1)
@njit
def Bivarcumnorm(a, b, r):
    a = np.asarray(a)
    b = np.asarray(b)
    r = np.asarray(r)
    N = len(a)
    
    result = np.zeros(N)
    r2 = np.zeros(N)

    # Gauss-Legendre 5-point quadrature nodes and weights
    x = np.array([0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992])
    W = np.array([0.018854042, 0.038088059, 0.0452707394, 0.038088059, 0.018854042])

    h1 = a
    h2 = b
    h12 = (h1**2 + h2**2) / 2

    #index1 = np.where(r >= 0.7)[0]
    #index2 = np.where(r < 0.7)[0]
    index1 = np.where((r >= 0.7) & np.isfinite(a) & np.isfinite(b))[0]
    index2 = np.where((r < 0.7) & np.isfinite(a) & np.isfinite(b))[0]

    # Case 1: r >= 0.7
    if len(index1) > 0:
        r_local = r[index1]
        r2[index1] = 1 - r_local**2
        r3 = np.sqrt(r2[index1])
        h1_ = h1[index1]
        h2_ = h2[index1]
        h3 = h1_ * h2_
        h7 = np.exp(-h3 / 2)
        h6 = np.abs(h1_ - h2_)
        h5 = h6**2 / 2
        h6 = h6 / r3
        AA = 0.5 - h3 / 8
        ab = 3 - 2 * AA * h5
        LH = 0.13298076 * h6 * ab * (1 - fast_norm_cdf(h6)) - \
             np.exp(-h5 / r2[index1]) * (ab + AA * r2[index1]) * 0.053051647
        
        for i in range(5):
            r1 = r3 * x[i]
            rr = r1**2
            r2_i = np.sqrt(1 - rr)
            h8 = np.exp(-h3 / (1 + r2_i)) / r2_i / h7
            LH -= W[i] * np.exp(-h5 / rr) * (h8 - 1 - AA * rr)

        result[index1] = LH * r3 * h7 + fast_norm_cdf(np.minimum(h1_, h2_))

    # Case 2: r < 0.7
    if len(index2) > 0:
        h1_ = h1[index2]
        h2_ = h2[index2]
        r_ = r[index2]
        h3 = h1_ * h2_
        LH = np.zeros_like(h1_)

        for i in range(5):
            r1 = r_ * x[i]
            r2_ = 1 - r1**2
            LH += W[i] * np.exp((r1 * h3 - h12[index2]) / r2_) / np.sqrt(r2_)

        result[index2] = fast_norm_cdf(h1_) * fast_norm_cdf(h2_) + r_ * LH

    # Boundary cases
    result[a == -np.inf] = 0
    result[b == -np.inf] = 0
    result[a == np.inf] = fast_norm_cdf(b[a == np.inf])
    result[b == np.inf] = fast_norm_cdf(a[b == np.inf])

    return result



#Fast implementation of weighted average
@njit
def fast_weighted_avg(x, w):
    total = 0.0
    wsum = 0.0
    for i in range(len(x)):
        total += x[i] * w[i]
        wsum += w[i]
    return total / wsum

#Fast implementation of sum of vector multiplication
@njit
def fast_vector_mult(x, w):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * w[i]
    return total

#Fast implementation of cumulative normal distribution
@njit
def fast_norm_cdf(x):
    result = np.empty(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        result[i] = 0.5 * (1.0 + erf(x[i] / sqrt(2.0)))
    return result



#Allocation of asset returns to rating classes
def Migration_events(Asset_return, Migration_boundaries_rating, Defaults):
    # Use searchsorted to assign ratings based on migration boundaries
    # Note: We reverse the boundaries to get correct indices from high to low
    Rating_allocation = np.searchsorted(Migration_boundaries_rating[::-1], Asset_return, side='right')
    
    # Convert to correct rating scale
    Rating_allocation = len(Migration_boundaries_rating) - Rating_allocation
    
    # Apply default overrides
    Rating_allocation[Defaults == 1] = len(Migration_boundaries_rating)
    
    return Rating_allocation

#Convert to numpy array if this is not already the case
def ensure_numpy(x):
    return x.to_numpy() if hasattr(x, 'to_numpy') else x

#Time format conversion for progress bar 
def format_eta(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h:{m:02d}m:{s:02d}s"