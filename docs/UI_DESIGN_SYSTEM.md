# Web UI design system

The implemented visual specification is [frontend/DESIGN.md](../frontend/DESIGN.md). Semantic tokens and responsive rules live in frontend/src/app/globals.css; terminal overrides follow the shared base rules.

## Direction

A dense, chart-led financial research terminal. Use flat adjoining panels, thin rules, compact command chrome and contextual readouts. Keep the main chart prominent and put secondary facts beside it. Avoid decorative card grids, gradients, glow, gauges and promotional typography. Completed daily-session data must say “not live”.

The default dark palette is charcoal/slate with restrained blue selection. Positive and negative colors describe measured changes; amber describes research and missing-data states. Text always carries the meaning. Research only, abstention and the failed V5 gate remain visible without repeated oversized warning banners.

## Actual tokens and scale

| Token | Dark | Light |
|---|---|---|
| Background | #101419 | #f4f6f9 |
| Surface | #151b22 | #ffffff |
| Raised | #1c252f | #eaf0f7 |
| Border | #303b48 | #c8d2e0 |
| Text | #e3e9ef | #18283b |
| Muted | #a1b0c0 | #536579 |
| Accent | #8bb8ec | #215fa6 |
| Accent soft | #21364c | #e0ecfa |
| Warning | #e7bb65 | #825600 |
| Danger | #ee9a9a | #a92e3a |
| Positive | #79c4a5 | #226e50 |
| Chart grid | #232c37 | #dbe3ee |

System sans with tabular numerals: body 13px, tables 11px, table headings 10px, page headings 20px, section headings 13px and quote values 24px. Small labels generally use 9–11px. Main panels, tables and charts have square corners; controls retain subtle 2–3px corners. CSS contains base rules followed by terminal composition overrides; the final matching rule determines the rendered value.

Desktop shell: 176px sidebar, 50px command bar and at least 28px context bar. Main padding is 18px 20px 32px, with maximum width 1800px. Use 18–22px gaps between workspaces; metric strips share separators rather than float as unrelated cards.

## Workspace patterns

- Dashboard: horizontal daily-session market strip, equity-focus chart with a 280px context rail, then sector/evidence matrices and guest watchlist/portfolio context. Collection details use progressive disclosure.
- Stock: shared InstrumentHeader, URL section tabs, chart plus instrument/forecast readout. Technical groups and detailed profile/risk/factor views remain contextual.
- Forecast: current completed close and saved origin are distinct. Adjacent horizon columns show archived estimates, ranges and abstention; diagnostics use AdvancedDetails. Evidence-stream selection never pools canonical and legacy records.
- Validate: 250px frozen-decision rail next to prospective counts, metric matrix and breakdowns. Historical V5 report is a separate section with the failed gate retained.
- Markets: grouped reference tables with a sector side column. Sector heatmap uses registered proxies and explicit availability, not invented breadth.
- Compare, Screener and Portfolio: 280px input rail beside results. Authenticated Python calculations run only on explicit submission. Watchlist uses editable browser guest positions and tracking results.
- Backtesting: local-operator internal blueprint with disabled execution; there is no strategy-job API or fabricated result.

## Disclosure, states and responsive behavior

Simple/Advanced is a persisted workspace preference. AdvancedDetails uses native details: Simple permits manual expansion; Advanced opens diagnostic sections. Navigation contains eight primary destinations plus More; protected research administration sits in the footer.

The sidebar becomes a modal drawer below 1000px. Chart/context and tool workspaces stack at 760px; mobile uses compact 12px body text at 540px, two-column horizons and scrollable dense tables. Main chart heights are typically 330–375px on desktop and 285px on small screens; large desktops increase the chart area. Do not promise real-time data or touch-target conformance from these dimensions alone.

Shared query states preserve loading, unavailable, degraded, error, pending and sealed semantics. Use “Unavailable”, never a synthetic zero. Search supports keyboard selection; focus outlines, skip link, dialog navigation, text chart context and table alternatives remain required. Optional transitions respect reduced motion. Final browser/build verification belongs in the completion report, not this design specification.
