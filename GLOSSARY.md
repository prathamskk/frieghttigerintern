# FreightTiger Shipping Cost Intelligence

Automated weekly cost tracking, comparison baselines, and context-grounded auditing for freight shipping costs across commercial transportation routes.

## Shipment & Network Metrics

**Route**:
A specific start and end point pair over which commercial freight shipments travel.
_Avoid_: Corridor, lane, track

**Route Type (`route_type`)**:
The distance classification (`Short`, `Medium`, or `Long`) assigned to a route.
_Avoid_: Distance bucket, distance tier, length classification

**Cost per Tonne per Km**:
Total freight cost divided by the product of cargo weight in tonnes and distance travelled in kilometers.
_Avoid_: Freight rate, unit shipping cost, tonne-km rate

**Week of (`week_of`)**:
The calendar Monday date designating a Monday-to-Sunday weekly aggregation period for shipment records.
_Avoid_: Shipment week, Monday cohort, week date, period

## Comparison Baselines

**vs. own history**:
A route's trailing 8-week rolling average cost per tonne-km, computed using only weeks strictly prior to the current week without look-ahead.
_Avoid_: Trailing baseline, historical average, own past average

**vs. similar routes**:
The average cost per tonne-km in that same week across all other routes sharing the same route type, excluding the route itself.
_Avoid_: Peer baseline, cohort average, route type average

**Cost Rise**:
A route-week where cost per tonne-km increases and looks out of the ordinary compared to its own history or similar routes.
_Avoid_: Candidate spike, anomaly, outlier, cost jump

## Audit & Explanation

**Context Note**:
A real-world operational note (such as fuel price changes, toll updates, weather disruptions, or festival surcharges) used to check whether a real reason explains a cost rise.
_Avoid_: Operational bulletin, incident report, external note

**Matched Note ID (`matched_note_id`)**:
The identifier of the specific context note verified to explain a cost rise, left blank when the rise is unexplained.
_Avoid_: Cited note, note reference, annotation ID

**Justified Verdict (`No (justified)`)**:
A verdict indicating a route's cost rise is supported by an applicable context note matching the route and time window, clearing the route from requiring human review.
_Avoid_: Cleared spike, explained anomaly, approved cost

**Unexplained Verdict (`Yes`)**:
A verdict indicating a route's cost rise has no valid supporting context note, flagged as "Yes" to require human review.
_Avoid_: Unjustified anomaly, positive flag, unexplained spike

**Reason (`reason`)**:
A plain-English explanation stating the verdict, reflecting the content and dates of the matched context note if justified, or explaining why the rise remains unexplained.
_Avoid_: Explanation narrative, causal summary, commentary

## Assistant

**Smart Assistant**:
An automated assistant that monitors weekly route costs, audits unexplained cost rises against context notes, and answers plain-English questions about its findings.
_Avoid_: Chatbot, conversational agent, LLM bot
