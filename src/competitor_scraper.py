import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import re
import json
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import date
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_theme, styled_fig, C

apply_theme()
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent

try:
    import requests
    from bs4 import BeautifulSoup
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


SEARCH_QUERIES: dict[str, str] = {
    "Figuras goma eva animales": "figuras goma eva animales",
    "Adornos de torta goma eva": "adorno torta goma eva",
    "Figuras Bichikids": "figuras bichikids goma eva",
    "Figuras de mono goma eva": "figuras mono goma eva",
    "Apliques goma eva": "apliques goma eva torta",
    "Carteles goma eva": "cartel decoracion goma eva",}


BUSINESS_PRICES: dict[str, float] = {
    "Figuras goma eva animales": 3500.0,   # median recent ticket for this category
    "Adornos de torta goma eva":  6500.0,
    "Figuras Bichikids": 4200.0,
    "Figuras de mono goma eva": 5000.0,
    "Apliques goma eva": 3200.0,
    "Carteles goma eva": 4500.0,}

# HTTP settings
HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MercadoLibre-pricing-research/1.0; "
        "contact: danielacordo24@gmail.com)"
    ),
    "Accept-Language": "es-AR,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml",}

ML_BASE_URL = "https://listado.mercadolibre.com.ar"
DELAY_MIN, DELAY_MAX = 2.0, 4.5   


@dataclass
class Listing:
    """A single competitor product listing from MercadoLibre search """
    title: str
    price: float
    seller: str
    url: str
    category: str
    scraped_date: str = field(default_factory=lambda: str(date.today()))


@dataclass
class PricePosition:
    """Business price position vs the market for one category."""
    category: str
    business_price:  float
    market_median: float
    market_p25: float
    market_p75: float
    market_min: float
    market_max: float
    n_competitors: int
    price_gap_pct: float    # (business - market_median) / market_median * 100
    position: str     
    recommendation: str



# Scraping
def _build_search_url(query: str, page: int = 1) -> str:
    """ Build a MercadoLibre Argentina search URL

    ML paginates by offset: page 1 = items 1-48, page 2 = items 49-96, etc."""
    slug = query.replace(" ", "-")
    offset = (page - 1) * 48
    if offset == 0:
        return f"{ML_BASE_URL}/{slug}"
    return f"{ML_BASE_URL}/{slug}_Desde_{offset + 1}_NoIndex_True"


def scrape_search( query: str, page: int = 1, session: Optional["requests.Session"] = None, timeout: int = 15,) -> Optional[str]:
    """ Fetch one MercadoLibre search results page and return the HTML.

    Notes:
    MercadoLibre uses anti-bot measures (Cloudflare, rate limiting).
    This function uses random delays, realistic headers, and session cookies """
    if not _REQUESTS_OK:
        raise ImportError(
            "requests and beautifulsoup4 are required for scraping. "
            "Install with: pip install requests beautifulsoup4")

    url = _build_search_url(query, page)
    sess = session or requests.Session()

    try:
        resp = sess.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:  # noqa: BLE001
        print(f" [WARN] Request failed for '{query}' page {page}: {e}")
        return None


def parse_listings(html: str, category: str) -> list[Listing]:
    """ Parses MercadoLibre search results HTML into Listing objects.

    ML changes its CSS class names frequently. This parser tries multiple selectors and falls back to regex on the raw HTML. """
    soup = BeautifulSoup(html, "html.parser")
    listings = []

    # Strategy 1: Structured JSON-LD (most reliable when present) 
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                items = data
            elif data.get("@type") in ("ItemList", "Product"):
                items = data.get("itemListElement", [data])
            else:
                continue
            for item in items:
                offer = item.get("offers", item.get("offer", {}))
                price = float(offer.get("price", 0))
                if price <= 0:
                    continue
                listings.append(Listing(
                    title = item.get("name", "")[:100],
                    price = price,
                    seller = offer.get("seller", {}).get("name", ""),
                    url = item.get("url", ""),
                    category = category,
                ))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    if listings:
        return listings

    # Strategy 2: CSS selectors (HTML parsing) 
    # ML uses different class names across A/B tests; try both old and new.
    price_selectors = [ "span.andes-money-amount__fraction", "span.price-tag-fraction", "[class*='price__fraction']",]
    title_selectors = [ "h2.ui-search-item__title", "h2.poly-box", "a.poly-component__title", "[class*='item__title']",]

    price_spans = []
    for sel in price_selectors:
        price_spans = soup.select(sel)
        if price_spans:
            break

    title_tags = []
    for sel in title_selectors:
        title_tags = soup.select(sel)
        if title_tags:
            break

    for title_tag, price_span in zip(title_tags, price_spans):
        try:
            raw_price = price_span.get_text().strip().replace(".", "").replace(",", ".")
            price = float(raw_price)
            if price <= 0:
                continue
            url = title_tag.get("href", "") if title_tag.name == "a" else ""
            listings.append(Listing(
                title = title_tag.get_text(strip=True)[:100],
                price = price,
                seller = "",
                url = url,
                category = category,
            ))
        except (ValueError, AttributeError):
            continue

    if listings:
        return listings

    # Strategy 3: Regex fallback on raw HTML 
    # ML embeds product data in a __PRELOADED_STATE__ JS variable as JSON.
    # This is the most robust approach but also the most brittle (JS bundle changes)
    match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.+?\});\s*</script>",
                      html, re.DOTALL)
    if match:
        try:
            state = json.loads(match.group(1))
            results = (state.get("initialState", {}).get("results", []))
            for item in results:
                price = item.get("price", 0)
                if price <= 0:
                    continue
                listings.append(Listing(
                    title = item.get("title", "")[:100],
                    price = float(price),
                    seller = item.get("seller", {}).get("nickname", ""),
                    url = item.get("permalink", ""),
                    category = category,))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    return listings


def scrape_category(category: str, query: str, pages:int = 2, session:Optional["requests.Session"] = None,) -> list[Listing]:
    """ Scrape multiple pages of search results for one product category """
    all_listings: list[Listing] = []
    seen_urls: set[str] = set()

    for page in range(1, pages + 1):
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        print(f" [{category}] page {page}/{pages} - waiting {delay:.1f}s ...", end=" ")

        time.sleep(delay)
        html = scrape_search(query, page=page, session=session)

        if html is None:
            print("FAILED")
            continue

        listings = parse_listings(html, category)
        print(f"{len(listings)} listings")

        for lst in listings:
            key = lst.url or f"{lst.title}|{lst.price}"
            if key not in seen_urls:
                seen_urls.add(key)
                all_listings.append(lst)

    return all_listings


def run_full_scrape(queries: dict[str, str] = SEARCH_QUERIES, pages:   int = 2,) -> pd.DataFrame:
    """ Scrape all categories and return a raw listings DataFrame """
    if not _REQUESTS_OK:
        raise ImportError("pip install requests beautifulsoup4 - required for live scraping")

    session = requests.Session()
    all_rows = []

    print(f"\nScraping {len(queries)} categories × {pages} pages each ...")
    for category, query in queries.items():
        print(f"\n Category: {category}")
        listings = scrape_category(category, query, pages=pages, session=session)
        for lst in listings:
            all_rows.append({
                "title": lst.title,
                "price": lst.price,
                "seller": lst.seller,
                "url": lst.url,
                "category": lst.category,
                "scraped_date": lst.scraped_date,})

    df = pd.DataFrame(all_rows)
    df = df[df["price"].between(100, 200_000)]  # sanity filter
    return df


# Synthetic Data (for testing without internet)
def generate_synthetic_snapshot(seed: int = 42) -> pd.DataFrame:
    """ Generates realistic synthetic competitor price data for testing.

    Based on actual MercadoLibre Argentina pricing patterns for goma eva decoration products (manual research, May 2026):
      - Price range: ~$1,500 – $25,000 ARS
      - Typical distribution: right-skewed (few premium outliers)
      - Market median by category varies 2x–3x
      - Business prices are ~10–30% above the market median
        (consistent with handmade/premium positioning)"""
    rng = np.random.default_rng(seed)

    # (category, market_median, n_competitors)
    categories = [
        ("Figuras goma eva animales", 2800, 35),
        ("Adornos de torta goma eva", 5200, 28),
        ("Figuras Bichikids", 3600, 18),
        ("Figuras de mono goma eva", 4100, 12),
        ("Apliques goma eva", 2500, 22),
        ("Carteles goma eva", 3800, 15),]

    rows = []
    today = str(date.today())

    for cat, median, n in categories:
        # Right-skewed lognormal distribution around the median
        mu = np.log(median)
        sigma = 0.45  # typical price spread in ARS marketplaces
        prices = rng.lognormal(mu, sigma, n)
        prices = np.clip(prices, median * 0.2, median * 4.5)

        for i, price in enumerate(prices):
            rows.append({
                "title": f"Producto {cat} #{i+1}",
                "price": round(float(price), 2),
                "seller": f"vendedor_{rng.integers(1000, 9999)}",
                "url": f"https://articulo.mercadolibre.com.ar/MLA-{rng.integers(1e8,1e9):.0f}",
                "category": cat,
                "scraped_date": today,})

    return pd.DataFrame(rows)


# Analysis
def compute_price_position(df: pd.DataFrame, business_prices: dict[str, float] = BUSINESS_PRICES,) -> list[PricePosition]:
    """ Compare business prices against the scraped market for each category"""
    positions = []

    for cat, biz_price in business_prices.items():
        market = df[df["category"] == cat]["price"]
        if len(market) < 3:
            continue

        median = market.median()
        gap = (biz_price - median) / median * 100

        if gap < -10:
            position = "below market"
            rec = (
                f"Price is {abs(gap):.0f}% below market median. "
                "Room to raise prices without losing competitive position."
            )
        elif gap < 5:
            position = "at market"
            rec = (
                f"Price is at market median (gap: {gap:+.0f}%). "
                "Safe to raise 10–15% — still within market range.")
        elif gap < 25:
            position = "above market"
            rec = (
                f"Price is {gap:.0f}% above median. "
                "Positioning as premium. Monitor conversion rate.")
        else:
            position = "premium"
            rec = (
                f"Price is {gap:.0f}% above median. "
                "Strong premium positioning - verify that product quality justifies the gap.")

        positions.append(PricePosition(
            category = cat,
            business_price = biz_price,
            market_median  = round(median, 0),
            market_p25 = round(market.quantile(0.25), 0),
            market_p75 = round(market.quantile(0.75), 0),
            market_min = round(market.min(), 0),
            market_max = round(market.max(), 0),
            n_competitors  = len(market),
            price_gap_pct  = round(gap, 1),
            position = position,
            recommendation = rec,))

    return positions


def estimate_cross_elasticity( snapshots: pd.DataFrame, own_sales: pd.DataFrame,) -> pd.DataFrame:
    """ Estimate cross-price elasticity: does the business's demand respond to changes in competitor prices?

    Model: log(Q_t) = alpha + epsilon_own * log(P_own_t) + epsilon_cross * log(P_competitors_t) + u_t

    Requirements:
    - snapshots: multi-period competitor price data (one row per scrape date x category)
    - own_sales: monthly order data for the same period """

    snapshots = snapshots.copy()
    snapshots["scraped_date"] = pd.to_datetime(snapshots["scraped_date"])
    n_dates = snapshots["scraped_date"].nunique()

    if n_dates < 6:
        return pd.DataFrame([{
            "Category": "All",
            "Status": "INSUFFICIENT DATA",
            "Explanation": (
                f"Cross-price elasticity requires competitor price data from at least "
                f"6 different dates (ideally monthly over 12+ months). "
                f"Only {n_dates} scrape date(s) available. "
                f"Run the scraper monthly and call this function again after 6 months."
            ),
            "Next step": (
                "Schedule: python src/competitor_scraper.py --save "
                "in cron or GitHub Actions, once per month."
            ),}])

    # Monthly competitor median per category
    snapshots["period"] = snapshots["scraped_date"].dt.to_period("M")
    comp_monthly = (snapshots.groupby(["category", "period"])["price"].median().reset_index().rename(columns={"price": "comp_median"}))

    # Monthly own volume
    own_sales = own_sales.copy()
    own_sales["period"] = pd.to_datetime(own_sales["Fecha"]).dt.to_period("M")
    own_monthly = (own_sales.groupby("period").agg(quantity=("Order_id", "count"), own_price=("Monto", "median")).reset_index())

    results = []
    for cat, grp in comp_monthly.groupby("category"):
        merged = grp.merge(own_monthly, on="period", how="inner")
        merged = merged[
            (merged["comp_median"] > 0)
            & (merged["quantity"] > 0)
            & (merged["own_price"] > 0)]
        if len(merged) < 6:
            continue

        log_q = np.log(merged["quantity"])
        log_p_own = np.log(merged["own_price"])
        log_p_com = np.log(merged["comp_median"])

        X = np.column_stack([np.ones(len(merged)), log_p_own, log_p_com])
        b = np.linalg.lstsq(X, log_q.values, rcond=None)[0]

        y_hat = X @ b
        ss_res = np.sum((log_q.values - y_hat) ** 2)
        ss_tot = np.sum((log_q.values - log_q.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        eps_own = round(b[1], 4)
        eps_cross = round(b[2], 4)

        if eps_cross > 0.2:
            cross_interp = (
                "POSITIVE cross-price elasticity: when competitors raise prices, "
                "this business's demand increases. Buyers are price-comparing - "
                "maintain competitive prices.")
        elif eps_cross > -0.1:
            cross_interp = (
                "NEAR-ZERO cross-price elasticity: this business's demand is largely "
                "independent of competitor prices. Unique product / loyal buyers - "
                "price increases are lower-risk.")
        else:
            cross_interp = (
                "NEGATIVE cross-price elasticity (unusual): higher competitor prices "
                "correlate with lower own demand. May indicate a shared demand driver. "
                "Investigate before acting.")

        results.append({
            "Category": cat,
            "epsilon_own": eps_own,
            "epsilon_cross": eps_cross,
            "R²": round(r2, 3),
            "n_periods": len(merged),
            "Cross-price interpretation": cross_interp,})

    return pd.DataFrame(results) if results else pd.DataFrame([{
        "Category": "All",
        "Status": "INSUFFICIENT OVERLAP",
        "Explanation": "No category had ≥ 6 matched periods of own + competitor data.",}])


def summarize_positions(positions: list[PricePosition]) -> pd.DataFrame:
    """Build a clean summary DataFrame from PricePosition results."""
    rows = []
    for p in positions:
        rows.append({
            "Category": p.category,
            "Business price": f"${p.business_price:,.0f}",
            "Market median": f"${p.market_median:,.0f}",
            "Market P25–P75": f"${p.market_p25:,.0f} – ${p.market_p75:,.0f}",
            "Gap vs median":  f"{p.price_gap_pct:+.1f}%",
            "Position": p.position,
            "Competitors (n)": p.n_competitors,
            "Recommendation": p.recommendation[:80],})
    return pd.DataFrame(rows)


# Visualization
def plot_competitive_landscape(df: pd.DataFrame, positions: list[PricePosition], save_path: Optional[Path] = None,) -> None:
    """ Two-panel chart: price distribution per category (violin + business marker) and price gap summary bar chart.

    Left panel: Violin plots of competitor price distributions per category.
                  Business price shown as a star marker.
    Right panel: Horizontal bar chart of price gap % vs market median.
                  Color-coded: green = below/at market, amber = above, red = premium."""
    n_cats = len(positions)
    if n_cats == 0:
        print("  No positions to plot.")
        return

    fig = styled_fig(18, 7,
        title="Competitive Price Intelligence - MercadoLibre Argentina",
        subtitle=(
            f"Competitor price distributions vs. business reference prices  -  "
            f"{df['scraped_date'].iloc[0] if 'scraped_date' in df.columns else 'latest snapshot'}  -  "
            f"{len(df)} listings across {n_cats} categories"
        ),)

    ax1 = fig.add_axes([0.04, 0.11, 0.56, 0.76])
    ax2 = fig.add_axes([0.66, 0.11, 0.31, 0.76])

    # Violin plots 
    cat_labels = [p.category for p in positions]
    data_by_cat = [
        df[df["category"] == p.category]["price"].values / 1000
        for p in positions]

    violin_parts = ax1.violinplot(
        data_by_cat,
        positions=range(n_cats),
        vert=False,
        showmedians=True,
        showextrema=True,)
    
    for pc in violin_parts["bodies"]:
        pc.set_facecolor(C["blue"])
        pc.set_alpha(0.35)
        pc.set_edgecolor(C["border"])
    violin_parts["cmedians"].set_color(C["accent"])
    violin_parts["cmedians"].set_linewidth(2)
    violin_parts["cbars"].set_color(C["muted"])
    violin_parts["cmins"].set_color(C["muted"])
    violin_parts["cmaxes"].set_color(C["muted"])

    # Business price markers (stars)
    for i, p in enumerate(positions):
        ax1.scatter(p.business_price / 1000, i, marker="*", s=220, color=C["gold"], zorder=6, label="Business price" if i == 0 else "",)

    ax1.set_yticks(range(n_cats))
    ax1.set_yticklabels([c[:30] for c in cat_labels], fontsize=8.5)
    ax1.set_xlabel("Price ($K ARS)")
    ax1.set_title("Price distribution vs. competitors\n(★ = business price, — = market median)", fontsize=11, pad=8)
    ax1.legend(fontsize=9, loc="lower right")

    # Gap bars
    gaps = [p.price_gap_pct for p in positions]
    y_pos = np.arange(n_cats)
    colors = []
    for g in gaps:
        if g < -10:
            colors.append(C["blue"])
        elif g < 5:
            colors.append(C["accent"])
        elif g < 25:
            colors.append(C["gold"])
        else:
            colors.append(C["red"])

    bars = ax2.barh(y_pos, gaps, height=0.55, color=colors, alpha=0.85, zorder=3)

    for i, (bar, p) in enumerate(zip(bars, positions)):
        x_text = bar.get_width()
        ha = "left" if x_text >= 0 else "right"
        offset = 0.5 if x_text >= 0 else -0.5
        ax2.text(x_text + offset, i, f"{p.price_gap_pct:+.0f}%", va="center", ha=ha, fontsize=8.5, color=C["text"], fontweight="bold")

    ax2.axvline(0, color=C["muted"], lw=1)
    ax2.axvline(10, color=C["gold"], lw=1, ls="--", alpha=0.5, label="Safe raise zone (+10%)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c[:22] for c in cat_labels], fontsize=8.5)
    ax2.set_xlabel("Price gap vs. market median (%)")
    ax2.set_title("Price gap\nvs. market median", fontsize=11, pad=8)
    ax2.legend(fontsize=8)

    # Color legend
    legend_items = [
        (C["blue"], "Below market (< -10%)"),
        (C["accent"], "At market (-10% to +5%)"),
        (C["gold"], "Above market (+5% to +25%)"),
        (C["red"], "Premium (> +25%)"),
    ]
    for k, (col, label) in enumerate(legend_items):
        ax2.text(0.02, 0.03 + k * 0.05, f"■ {label}", transform=ax2.transAxes, fontsize=7, color=col, va="bottom",)

    fig.text(0.5, 0.02, "Data: live MercadoLibre Argentina search results. "
        "Prices are list prices - actual transaction prices may differ after discounts. "
        "Re-scrape monthly to track market movements.", ha="center", fontsize=8.5, color=C["muted"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor=C["surf2"], edgecolor=C["border"]),)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# Main
def run_competitor_analysis(snapshot_path: Optional[Path] = None, dry_run: bool = False,
    save_snapshot: bool = True, plots_dir: Optional[Path] = None, pages: int = 2,  quiet: bool = False,) -> dict:
    """ Run full competitor price intelligence pipeline """

    # Step 1: Get competitor data 
    if dry_run:
        if not quiet:
            print("\n[DRY RUN] Using synthetic competitor price data ...")
        df = generate_synthetic_snapshot()
    elif snapshot_path and Path(snapshot_path).exists():
        if not quiet:
            print(f"\nLoading saved snapshot: {snapshot_path}")
        df = pd.read_csv(snapshot_path)
    else:
        if not quiet:
            print("\nStarting live scrape of MercadoLibre Argentina ...")
            print("(Run with --dry-run to use synthetic data instead)\n")
        df = run_full_scrape(pages=pages)

    if df.empty:
        print(" [ERROR] No competitor data available. Use --dry-run to test.")
        return {}

    # Save snapshot
    if save_snapshot and not dry_run:
        out_path = BASE / "data" / "competitor_prices.csv"
        df.to_csv(out_path, index=False)
        if not quiet:
            print(f"\n Snapshot saved: {out_path}")

    # Step 2: Price position analysis 
    positions = compute_price_position(df)
    positions_df = summarize_positions(positions)

    # Step 3: Cross-price elasticity 
    own_sales = pd.read_csv(BASE / "data" / "ventas_decoraciones.csv", parse_dates=["Fecha"])
    cross_elast = estimate_cross_elasticity(df, own_sales)

    # Output 
    if not quiet:
        print("\n")
        print("COMPETITOR PRICE INTELLIGENCE")
        print()
        print(f"\n Listings collected: {len(df)}")
        print(f" Categories: {df['category'].nunique()}")
        print(f" Date: {df['scraped_date'].iloc[0]}")

        print("\n Price position vs. market")
        print(positions_df.to_string(index=False))

        print("\n Cross-price elasticity")
        print(cross_elast.to_string(index=False))

        print("\n Strategic implications")
        above = [p for p in positions if "above" in p.position or "premium" in p.position]
        below = [p for p in positions if "below" in p.position]
        at = [p for p in positions if p.position == "at market"]

        if below:
            print(f"\n  ↑ PRICE INCREASE OPPORTUNITY ({len(below)} categories below market):")
            for p in below:
                print(f" - {p.category}: business ${p.business_price:,.0f} "
                      f"vs market median ${p.market_median:,.0f} ({p.price_gap_pct:+.0f}%)")
        if at:
            print(f"\n  -> AT MARKET ({len(at)} categories):")
            for p in at:
                print(f" - {p.category}: at median, safe to test +10%")
        if above:
            print(f"\n PREMIUM POSITIONING ({len(above)} categories above market):")
            for p in above:
                print(f" - {p.category}: {p.price_gap_pct:+.0f}% above median "
                      f"monitor conversion")

    # Visualization 
    if plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_competitive_landscape(
            df = df,
            positions = positions,
            save_path = plots_dir / "competitor_prices.png",)

    return {
        "snapshot_df": df,
        "positions": positions,
        "positions_df": positions_df,
        "cross_elasticity": cross_elast,}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competitor price scraper and analysis")
    parser.add_argument("--snapshot", default=None, help="Path to saved CSV snapshot (skips scraping)")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data (no internet required)")
    parser.add_argument("--pages", type=int, default=2, help="Pages to scrape per category (default 2 = ~96 listings)")
    parser.add_argument("--no-save", action="store_true", help="Do not save snapshot to data/")
    parser.add_argument("--plots", default=str(BASE / "plots"), help="Directory to save plots")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_competitor_analysis(
        snapshot_path = args.snapshot,
        dry_run = args.dry_run,
        save_snapshot = not args.no_save,
        plots_dir = Path(args.plots),
        pages = args.pages,
        quiet = args.quiet,)
