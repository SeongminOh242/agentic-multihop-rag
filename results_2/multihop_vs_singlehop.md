# Multi-hop vs Single-hop Comparison

This report splits validation questions by whether the **gold supporting facts** span multiple Wikipedia titles.

## Subset metrics (agent only)

| Subset | Count | EM | F1 | Avg hops used |
|---|---:|---:|---:|---:|
| gold single-hop (1 title) | 0 | 0.000 | 0.000 | 0.00 |
| gold multi-hop (2+ titles) | 50 | 0.120 | 0.209 | 2.00 |

## Example traces (gold single-hop)

_No examples._

## Example traces (gold multi-hop)

- **id**: `5abd94525542992ac4f382d2`
  - **question**: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
  - **gold_answer**: YG Entertainment
  - **agent_answer**: YG Entertainment
  - **gold_supporting_titles**: 2014 S/S, Winner (band)
  - **agent_hops**: 2
    - **hop_query**: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
      - **top_titles**: List of awards and nominations received by Shinee, 2014 S/S, List of songs written by Ravi, BTS discography, Madtown
    - **hop_query**: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
      - **top_titles**: List of awards and nominations received by Shinee, 2014 S/S, List of songs written by Ravi, BTS discography, Madtown
- **id**: `5ae2070a5542994d89d5b313`
  - **question**: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 
  - **gold_answer**: Badly Drawn Boy
  - **agent_answer**: Badly Drawn Boy
  - **gold_supporting_titles**: Badly Drawn Boy, Wolf Alice
  - **agent_hops**: 2
    - **hop_query**: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 
      - **top_titles**: Badly Drawn Boy discography, Something to Talk About (Badly Drawn Boy song), About a Boy (soundtrack), The Hour of Bewilderbeast, Under an Hour
    - **hop_query**: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 
      - **top_titles**: Badly Drawn Boy discography, Something to Talk About (Badly Drawn Boy song), About a Boy (soundtrack), The Hour of Bewilderbeast, Under an Hour
- **id**: `5a828c8355429966c78a6a50`
  - **question**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?
  - **gold_answer**: Henry J. Kaiser
  - **agent_answer**: Henry J. Kaiser
  - **gold_supporting_titles**: Henry J. Kaiser, Kaiser Ventures
  - **agent_hops**: 2
    - **hop_query**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?
      - **top_titles**: Henry J, Kaiser Ventures, Neafie &amp; Levy, Kaiser Shipyards, Edgar Kaiser Jr
    - **hop_query**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?
      - **top_titles**: Henry J, Kaiser Ventures, Neafie &amp; Levy, Kaiser Shipyards, Edgar Kaiser Jr
