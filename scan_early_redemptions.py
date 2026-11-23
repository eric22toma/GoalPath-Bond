"""One-time AGG premature data disappearance audit for select bonds.
This file cross-references zipped AGG observations against WRDS callable-bond data
to flag bonds that may have been redeemed early. Flagged bonds 
were reviewed against SEC filings and the FINRA fixed income site.

Confirmed early redemptions are grouped in EARLY_REDEMPTIONS, and other
reviewed bonds in NOT_REDEEMED_EARLY. Both are excluded from future scans.
New candidates are printed to the terminal with priorities 1, 2, and 3.

Terminal-return calculations remain commented out for possible future
investigations. This diagnostic scanner is not part of the backtest pipeline.
"""

from pathlib import Path
import pandas as pd

from process_investable_universe import DEFAULT_BONDS, PROJECT_ROOT

REFERENCE = PROJECT_ROOT / '01_Source_Data' / 'early_redemption_reference.csv'
START_MONTH = '2020-12'  # Set as this date to match the current backtest decision-month window.
END_MONTH = '2025-10'

'''Early redemptions: CUSIP -> (redemption date, price per $100 face).
Preserve exact FINRA/issuer redemption dates where available; otherwise
use calendar month-end given all other datasets follow this convention.
Note that redeemed bonds without a provided redemption price were defauted 
to face value rdemptions of $100 per $100 face value. '''

EARLY_REDEMPTIONS = {
    '876030AD9': ('2024-11-25', 101.0),  # Tapestry Inc; issuer/SEC-reported date; not independently checked against FINRA
    '876030AE7': ('2024-11-25', 101.0),  # Tapestry Inc; issuer/SEC-reported date; not independently checked against FINRA
    '876030AF4': ('2024-11-25', 101.0),  # Tapestry Inc; issuer/SEC-reported date; not independently checked against FINRA
    '501044DS7': ('2024-12-18', 101.0),  # Kroger Co; issuer/SEC-reported date; not independently checked against FINRA
    '501044DT5': ('2024-12-18', 101.0),  # Kroger Co; issuer/SEC-reported date; not independently checked against FINRA
    '501044DU2': ('2024-12-18', 101.0),  # Kroger Co; issuer/SEC-reported date; not independently checked against FINRA
    '76132FAC1': ('2025-02-22', 100.0),  # Retail Opportunity Investments Partnership LP; date retained from prior audit; official FINRA source not documented
    '595112BP7': ('2025-05-27', 100.0),  # Micron Technology Inc; date retained from prior audit; official FINRA source not documented
    '46590XAZ9': ('2025-07-03', 102.478),  # JBS USA / JBS USA Finance; date retained from prior audit; official FINRA source not documented
    '929160BA6': ('2024-03-01', 100.0),  # Vulcan Materials Co; issuer/SEC-reported date; not independently checked against FINRA
    '750236AX9': ('2024-03-19', 100.396),  # Radian Group Inc; issuer/SEC-reported date; not independently checked against FINRA
    '91911TAP8': ('2024-07-25', 100.0),  # Vale Overseas Ltd; issuer/SEC-reported date; not independently checked against FINRA
    '595112BV4': ('2025-10-24', 100.0),  # Micron Technology Inc; issuer/SEC-reported date; not independently checked against FINRA
    '595112CA9': ('2025-10-24', 100.0),  # Micron Technology Inc; issuer/SEC-reported date; not independently checked against FINRA
    '12673PAJ4': ('2025-10-23', 100.0),  # CA Inc; date retained from prior audit; official FINRA source not documented
    '11134LAH2': ('2025-10-23', 100.0),  # Broadcom Corp / Broadcom Cayman Finance Ltd; date retained from prior audit; official FINRA source not documented
    '928563AE5': ('2025-10-03', 100.0),  # VMware Inc; date retained from prior audit; official FINRA source not documented
    '87612BBL5': ('2025-07-15', 100.0),  # Targa Resources Partners LP; date retained from prior audit; official FINRA source not documented
    '256677AE5': ('2025-09-30', 100.0),  # Dollar General Corp; month-end modelling proxy; not a confirmed FINRA date
    '23317HAF9': ('2024-08-21', 100.0),  # DDR Corp (SITE Centers); date retained from prior audit; official FINRA source not documented
    '81618TAE0': ('2024-03-31', 100.0),  # Select Income REIT; date retained from prior audit; official FINRA source not documented
    '04273WAD3': ('2024-04-30', 100.0),  # Arrow Electronics Inc; date retained from prior audit; official FINRA source not documented
    '16412XAC9': ('2024-04-30', 100.0),  # Cheniere Energy Inc; date retained from prior audit; official FINRA source not documented
    '14162VAB2': ('2025-07-31', 100.0),  # Care Capital Properties LP; date retained from prior audit; official FINRA source not documented
    '29250NBQ7': ('2024-03-08', 100.0),  # Enbridge Inc; date retained from prior audit; official FINRA source not documented
    '462613AM2': ('2024-04-30', 100.0),  # IPALCO Enterprises Inc; date retained from prior audit; official FINRA source not documented
    '14448CBB9': ('2024-06-30', 100.0),  # Carrier Global Corp; date retained from prior audit; official FINRA source not documented
    '29364WBK3': ('2024-08-31', 100.0),  # Entergy Louisiana LLC; date retained from prior audit; official FINRA source not documented
    '581557BS3': ('2024-09-30', 100.0),  # McKesson Corp; month-end modelling proxy; not a confirmed FINRA date
    '42824CBM0': ('2024-10-31', 100.0),  # Hewlett Packard Enterprise Co; date retained from prior audit; official FINRA source not documented
    '71429MAB1': ('2024-10-02', 100.0),  # Perrigo Finance Unlimited Co; date retained from prior audit; official FINRA source not documented
    '444859BW1': ('2024-11-30', 100.0),  # Humana Inc; date retained from prior audit; official FINRA source not documented
    '559080AK2': ('2025-12-31', 100.0),  # Magellan Midstream Partners LP; month-end modelling proxy; not a confirmed FINRA date
    '844741BJ6': ('2024-12-31', 100.0),  # Southwest Airlines Co; date retained from prior audit; official FINRA source not documented
    '92556HAA5': ('2024-12-27', 100.0),  # ViacomCBS Inc (Paramount Global); date retained from prior audit; official FINRA source not documented
    '92940PAB0': ('2024-12-06', 100.0),  # WRKCo Inc (WestRock); date retained from prior audit; official FINRA source not documented
    '370334CS1': ('2025-02-28', 100.0),  # General Mills Inc; month-end modelling proxy; not a confirmed FINRA date
    '361841AJ8': ('2025-03-03', 100.0),  # GLP Capital LP / GLP Financing II Inc; date retained from prior audit; official FINRA source not documented
    '436106AA6': ('2025-02-16', 100.0),  # HollyFrontier Corp; date retained from prior audit; official FINRA source not documented
    '53522KAB9': ('2025-02-28', 100.0),  # Linde Inc; date retained from prior audit; official FINRA source not documented
    '55903VBG7': ('2025-02-28', 100.0),  # WarnerMedia Holdings Inc; date retained from prior audit; official FINRA source not documented
    '595112BM4': ('2025-02-28', 100.0),  # Micron Technology Inc; date retained from prior audit; official FINRA source not documented
    '636180BQ3': ('2025-03-06', 100.0),  # National Fuel Gas Co; date retained from prior audit; official FINRA source not documented
    '876030AB3': ('2024-11-25', 101.0),  # Tapestry Inc; issuer/SEC-reported date; not independently checked against FINRA
    '876030AC1': ('2024-11-25', 101.0),  # Tapestry Inc; issuer/SEC-reported date; not independently checked against FINRA
    '501044DR9': ('2024-12-18', 101.0),  # Kroger Co; issuer/SEC-reported date; not independently checked against FINRA
    '24703TAC0': ('2024-10-11', 100.0),  # Dell International LLC / EMC Corp; date confirmed via official FINRA site (supplied by user)
    '95001DCG3': ('2024-10-28', 100.0),  # Wells Fargo & Co; date confirmed via official FINRA site (supplied by user)
    '00206RMP4': ('2024-11-09', 100.0),  # AT&T Inc; date confirmed via official FINRA site (supplied by user)
    '31572UAG1': ('2024-11-14', 100.0),  # Fibria Overseas Finance Ltd; date confirmed via official FINRA site (supplied by user)
    '097023CT0': ('2024-12-20', 100.0),  # Boeing Co; date confirmed via official FINRA site (supplied by user)
    '025816CQ0': ('2025-02-03', 100.0),  # American Express Co; date confirmed via official FINRA site (supplied by user)
    '035240AL4': ('2025-05-30', 100),  # Anheuser-Busch InBev Worldwide Inc; date retained from prior audit; official FINRA source not documented
    '866677AK3': ('2025-05-10', 106.153),  # Sun Communities Operating LP; date retained from prior audit; official FINRA source not documented
    '866677AJ6': ('2025-05-10', 108.588),  # Sun Communities Operating LP; date retained from prior audit; official FINRA source not documented
    '74348TAU6': ('2025-06-18', 100.0),  # Prospect Capital Corp; date retained from prior audit; official FINRA source not documented
    '026874DH7': ('2025-06-26', 100.0),  # American International Group Inc; date retained from prior audit; official FINRA source not documented
    '21036PBN7': ('2025-06-12', 100.0),  # Constellation Brands Inc; date retained from prior audit; official FINRA source not documented
    '52107QAH8': ('2025-09-18', 100.0),  # Lazard Group LLC; date retained from prior audit; official FINRA source not documented
    '55336VAJ9': ('2025-04-09', 100.0),  # MPLX LP; date retained from prior audit; official FINRA source not documented
    '674599EE1': ('2025-03-31', 100.0),  # Occidental Petroleum Corp; month-end modelling proxy; not a confirmed FINRA date
    '674599EB7': ('2025-06-30', 100.0),  # Occidental Petroleum Corp; month-end modelling proxy; not a confirmed FINRA date
    '674599DC6': ('2025-06-30', 100.0),  # Occidental Petroleum Corp; month-end modelling proxy; not a confirmed FINRA date
    '92343VDD3': ('2025-04-23', 100.0),  # Verizon Communications Inc; date retained from prior audit; official FINRA source not documented
    '559222AX2': ('2025-05-23', 100.0),  # Magna International Inc; date retained from prior audit; official FINRA source not documented
    '65163LAQ2': ('2025-02-07', 100.0),  # Newmont Corp / Newcrest Finance Pty Ltd; date retained from prior audit; official FINRA source not documented
    '125523CR9': ('2025-03-31', 100.0),  # The Cigna Group; month-end modelling proxy; not a confirmed FINRA date
    '302491AW5': ('2025-06-30', 100.0),  # FMC Corp; month-end modelling proxy; not a confirmed FINRA date
    '651229AW6': ('2025-05-31', 100.0),  # Newell Rubbermaid Inc; date retained from prior audit; official FINRA source not documented
    '00751YAH9': ('2025-10-31', 100.0),  # Advance Auto Parts Inc; month-end modelling proxy; not a confirmed FINRA date
    '031162DN7': ('2025-12-31', 100.0),  # Amgen Inc; month-end modelling proxy; not a confirmed FINRA date
    '95000U2K8': ('2025-02-11', 100.0),  # Wells Fargo & Co; date confirmed via official FINRA site (supplied by user)
    '198280AF6': ('2025-03-24', 100.0),  # Columbia Pipeline Group Inc; date confirmed via official FINRA site (supplied by user)
    '26885BAC4': ('2025-05-01', 100.0),  # EQM Midstream Partners LP; date confirmed via official FINRA site (supplied by user)
    '025816CY3': ('2025-07-01', 100.0),  # American Express Co; date confirmed via official FINRA site (supplied by user)
    '81180WAR2': ('2025-06-11', 100.0),  # Seagate HDD Cayman; date confirmed via official FINRA site (supplied by user)
    '37045VAV2': ('2025-09-01', 100.0),  # General Motors Corp; date confirmed via official FINRA site (supplied by user)
    '403949AB6': ('2025-08-30', 100.0),  # HF Sinclair Corp; date confirmed via official FINRA site (supplied by user)
    '403949AP5': ('2025-08-31', 100.0),  # HF Sinclair Corp; month-end modelling proxy; not a confirmed FINRA date
    '44106MAV4': ('2025-09-04', 100.0),  # Hospitality Properties Trust (Service Properties Trust); date confirmed via official FINRA site (supplied by user)
    '87264AAT2': ('2025-09-01', 100.0),  # T-Mobile USA Inc; date confirmed via official FINRA site (supplied by user)
    '98978VAU7': ('2025-08-28', 100.0),  # Zoetis Inc; date confirmed via official FINRA site (supplied by user)
    '023135BN5': ('2025-09-05', 100.0),  # Amazon.com Inc; date confirmed via official FINRA site (supplied by user)
    '28622HAA9': ('2025-09-19', 100.0),  # Elevance Health Inc; date confirmed via official FINRA site (supplied by user)
    '361841AH2': ('2025-09-12', 100.0),  # GLP Capital LP / GLP Financing II Inc; date confirmed via official FINRA site (supplied by user)
    '674599CH6': ('2025-09-30', 100.0),  # Occidental Petroleum Corp; date confirmed via official FINRA site (supplied by user)
    '931427AQ1': ('2025-09-30', 100.0),  # Walgreens Boots Alliance Inc; month-end modelling proxy; not a confirmed FINRA date
    '966837AE6': ('2025-09-30', 100.0),  # Whole Foods Market Inc; month-end modelling proxy; not a confirmed FINRA date
    '85207UAK1': ('2025-10-31', 100.0),  # Sprint Corp; month-end modelling proxy; not a confirmed FINRA date
    '302491AS4': ('2023-12-21', 100.0),  # FMC Corp; issuer/SEC-reported date; not independently checked against FINRA
    '44106MAS1': ('2023-12-09', 100.0),  # Hospitality Properties Trust (Service Properties Trust); issuer/SEC-reported date; not independently checked against FINRA
    '81721MAK5': ('2024-01-31', 100.0),  # Diversified Healthcare Trust; month-end modelling proxy; not a confirmed FINRA date
    '559080AH9': ('2024-02-29', 100.0),  # Magellan Midstream Partners LP; month-end modelling proxy; not a confirmed FINRA date
    '013817AW1': ('2024-07-01', 100.0),  # Howmet Aerospace Inc; issuer/SEC-reported date; not independently checked against FINRA
    '105340AN3': ('2024-06-07', 100.0),  # Brandywine Operating Partnership LP; issuer/SEC-reported date; not independently checked against FINRA
    '887389AJ3': ('2024-06-24', 100.0),  # Timken Co; issuer/SEC-reported date; not independently checked against FINRA
    '896522AH2': ('2024-06-25', 100.0),  # Trinity Industries Inc; issuer/SEC-reported date; not independently checked against FINRA
    '01959EAC2': ('2024-07-24', 100.0),  # Allied World Assurance Holdings; issuer/SEC-reported date; not independently checked against FINRA
    '202795JE1': ('2024-07-31', 100.0),  # Commonwealth Edison Co; month-end modelling proxy; not a confirmed FINRA date
    '23317HAD4': ('2024-08-23', 100.0),  # SITE Centers Corp; issuer/SEC-reported date; not independently checked against FINRA
    '23317HAE2': ('2024-08-23', 99.4),  # SITE Centers Corp; issuer/SEC-reported date; not independently checked against FINRA
    '256677AD7': ('2025-04-29', 100),  # Dollar General Corp; date confirmed via official FINRA site (supplied by user)
    '174610AJ4': ('2025-07-03', 100),  # Citizens Financial Group Inc; issuer/SEC-reported date; not independently checked against FINRA
    '21036PAP3': ('2025-07-02', 100),  # Constellation Brands Inc; issuer/SEC-reported date; not independently checked against FINRA
    '88947EAR1': ('2025-07-15', 100),  # Toll Brothers Finance Corp; date confirmed via official FINRA site (supplied by user)
    '06652KAA1': ('2025-08-22', 100),  # BankUnited Inc; issuer/SEC-reported date; not independently checked against FINRA
    '25179MAV5': ('2025-09-15', 100),  # Devon Energy Corp; date confirmed via official FINRA site (supplied by user)
    '86765BAS8': ('2025-09-22', 100),  # Sunoco Logistics Partners Operations LP (Energy Transfer); date confirmed via official FINRA site (supplied by user)
    '843646AM2': ('2025-10-21', 100),  # Southern Power Co; date confirmed via official FINRA site (supplied by user)
    '681936BH2': ('2025-10-15', 100),  # Omega Healthcare Investors Inc; date confirmed via official FINRA site (supplied by user)
    '174610AK1': ('2025-11-03', 100),  # Citizens Financial Group Inc; issuer/SEC-reported date; not independently checked against FINRA
    '92345YAD8': ('2025-04-21', 100),  # Verisk Analytics Inc; date confirmed via official FINRA site (supplied by user)
    '12505BAD2': ('2025-05-28', 100.17),  # CBRE Services Inc; date confirmed via official FINRA site (supplied by user)
    '053015AE3': ('2025-06-15', 100),  # Automatic Data Processing Inc; date confirmed via official FINRA site (supplied by user)
    '09062XAF0': ('2025-06-27', 100),  # Biogen Inc; date confirmed via official FINRA site (supplied by user)
    '20030NBN0': ('2025-06-05', 100),  # Comcast Corp; date confirmed via official FINRA site (supplied by user)
    '29250RAW6': ('2025-07-28', 100),  # Enbridge Energy Partners LP; date confirmed via official FINRA site (supplied by user)
    '931427AB4': ('2025-09-29', 100),  # Walgreens Boots Alliance Inc; date confirmed via official FINRA site (supplied by user)
    '931427AC2': ('2025-09-29', 100),  # Walgreens Boots Alliance Inc; redemption date not independently confirmed on official FINRA site
    '98978VAK9': ('2025-09-17', 100),  # Zoetis Inc; date confirmed via official FINRA site (supplied by user)
    '58013MEY6': ('2025-11-03', 100),  # McDonald's Corp; date confirmed via official FINRA site (supplied by user)
    '279158AL3': ('2024-11-21', 100.763),  # Ecopetrol SA; issuer/SEC-reported date; not independently checked against FINRA
    '47102XAJ4': ('2024-11-07', 100),  # Janus Henderson US Holdings Inc; issuer/SEC-reported date; not independently checked against FINRA
    '85207UAJ4': ('2024-11-15', 100),  # Sprint Corp; issuer/SEC-reported date; not independently checked against FINRA
    '674599CG8': ('2024-12-30', 100),  # Occidental Petroleum Corp; date confirmed via official FINRA site (supplied by user)
    '68268NAP8': ('2024-12-15', 100),  # ONEOK Partners LP; date confirmed via official FINRA site (supplied by user)
    '898813AP5': ('2024-12-16', 100),  # Tucson Electric Power Co; date confirmed via official FINRA site (supplied by user)
    '025816BK4': ('2024-11-04', 100),  # American Express Co; date confirmed via official FINRA site (supplied by user)
    '00037BAE2': ('2020-12-31', 100),  # ABB Finance USA Inc; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '035242AL0': ('2020-12-31', 100),  # Anheuser-Busch InBev Finance Inc; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '05531FAZ6': ('2020-12-31', 100),  # BB&T Corp; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '78355HKC2': ('2020-12-31', 100),  # Ryder System Inc; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '026874DG9': ('2021-02-28', 100),  # American International Group Inc; user-confirmed redemption, 2021-02 month-end proxy (exact day not supplied)
    '151020BA1': ('2021-03-31', 100.15),  # Celgene Corp; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '460146CP6': ('2021-10-31', 100),  # International Paper Co; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '31620MAT3': ('2021-04-30', 100),  # Fidelity National Information Services Inc; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '58507LAC3': ('2022-05-31', 100.15),  # Medtronic Global Holdings SCA; user-confirmed redemption, 2022-05 month-end proxy (exact day not supplied)
    '035240AJ9': ('2020-12-31', 100),  # Anheuser-Busch InBev Worldwide Inc; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '44106MAU6': ('2020-12-31', 100),  # Hospitality Properties Trust; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '7591EPAM2': ('2020-12-31', 100),  # Regions Financial Corp; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '88579YAU5': ('2020-12-31', 100),  # 3M Co; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '88579YBA8': ('2020-12-31', 100),  # 3M Co; user-confirmed redemption, 2020-12 month-end proxy (exact day not supplied)
    '285512AC3': ('2021-02-28', 100),  # Electronic Arts Inc; user-confirmed redemption, 2021-02 month-end proxy (exact day not supplied)
    '37045XBF2': ('2021-02-28', 100),  # General Motors Financial Co Inc; user-confirmed redemption, 2021-02 month-end proxy (exact day not supplied)
    '82481LAB5': ('2021-01-31', 100),  # Shire Acquisitions Investments Ireland DAC; user-confirmed redemption, 2021-01 month-end proxy (exact day not supplied)
    '828807CX3': ('2021-01-31', 100),  # Simon Property Group LP; user-confirmed redemption, 2021-01 month-end proxy (exact day not supplied)
    '88160QAN3': ('2021-01-31', 100),  # Tesoro Logistics LP / Tesoro Logistics Finance Corp; user-confirmed redemption, 2021-01 month-end proxy (exact day not supplied)
    '883556BN1': ('2021-01-31', 100),  # Thermo Fisher Scientific Inc; user-confirmed redemption, 2021-01 month-end proxy (exact day not supplied)
    '12673PAH8': ('2021-02-28', 100),  # CA Inc; user-confirmed redemption, 2021-02 month-end proxy (exact day not supplied)
    '25389JAS5': ('2021-02-28', 104.71),  # Digital Realty Trust LP; user-confirmed redemption, 2021-02 month-end proxy (exact day not supplied)
    '373334KF7': ('2021-03-31', 100),  # Georgia Power Co; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '651229AU0': ('2021-03-31', 100),  # Newell Rubbermaid Inc; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '724479AK6': ('2021-02-28', 100),  # Pitney Bowes Inc; user-confirmed redemption, 2021-02 month-end proxy (exact day not supplied)
    '891906AD1': ('2021-03-31', 100),  # Total System Services Inc; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '015271AM1': ('2021-03-31', 100),  # Alexandria Real Estate Equities Inc; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '124857AS2': ('2021-03-31', 100),  # CBS Corp; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '26138EAV1': ('2021-03-31', 100),  # Dr Pepper Snapple Group Inc; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '29736RAH3': ('2021-04-30', 100),  # Estée Lauder Companies Inc; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '574599BL9': ('2021-03-31', 100),  # Masco Corp; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '666807BQ4': ('2021-03-31', 100),  # Northrop Grumman Corp; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '87854XAD3': ('2021-03-31', 100),  # TechnipFMC PLC; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '92277GAK3': ('2021-03-31', 104.88),  # Ventas Realty LP; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '959802AU3': ('2021-04-30', 100),  # Western Union Co; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '053807AT0': ('2021-05-31', 101.67),  # Avnet Inc; user-confirmed redemption, 2021-05 month-end proxy (exact day not supplied)
    '06406FAB9': ('2021-04-30', 100),  # Bank of New York Mellon Corp; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '0258M0EB1': ('2021-04-30', 100),  # American Express Credit Corp; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '88579YBF7': ('2021-03-31', 100),  # 3M Co; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '437076BL5': ('2021-03-31', 100.15),  # Home Depot Inc; user-confirmed redemption, 2021-03 month-end proxy (exact day not supplied)
    '15189XAP3': ('2021-05-31', 100),  # CenterPoint Energy Houston Electric LLC; user-confirmed redemption, 2021-05 month-end proxy (exact day not supplied)
    '38141GWC4': ('2021-04-30', 100),  # Goldman Sachs Group Inc; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '05531FBD4': ('2021-08-31', 100),  # BB&T Corp; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '21036PAR9': ('2021-08-31', 100),  # Constellation Brands Inc; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '21036PAW8': ('2021-08-31', 100),  # Constellation Brands Inc; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '478375AF5': ('2021-09-30', 100),  # Johnson Controls International PLC; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '7591EPAP5': ('2021-08-31', 100),  # Regions Financial Corp; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '294429AK1': ('2021-05-31', 100),  # Equifax Inc; user-confirmed redemption, 2021-05 month-end proxy (exact day not supplied)
    '842587CS4': ('2021-06-30', 100),  # Southern Co; user-confirmed redemption, 2021-06 month-end proxy (exact day not supplied)
    '037833CC2': ('2021-07-31', 100),  # Apple Inc; user-confirmed redemption, 2021-07 month-end proxy (exact day not supplied)
    '14042RBT7': ('2021-08-31', 100),  # Capital One NA; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '50077LAS5': ('2021-05-31', 100),  # Kraft Heinz Foods Co; user-confirmed redemption, 2021-05 month-end proxy (exact day not supplied)
    '50077LAK2': ('2021-07-31', 100),  # Kraft Heinz Foods Co; user-confirmed redemption, 2021-07 month-end proxy (exact day not supplied)
    '233331BA4': ('2023-07-31', 100),  # DTE Energy Co; user-confirmed redemption, 2023-07 month-end proxy (exact day not supplied); dataset last observed 2021-06, earlier exit remains unexplained
    '233331AV9': ('2022-04-30', 100),  # DTE Energy Co; user-confirmed redemption, 2022-04 month-end proxy (exact day not supplied); dataset last observed 2021-06, earlier exit remains unexplained
    '075887BT5': ('2021-09-30', 100),  # Becton Dickinson & Co; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '724479AL4': ('2021-09-30', 100),  # Pitney Bowes Inc; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '05565QDP0': ('2021-07-31', 100.125),  # BP Capital Markets PLC; user-confirmed redemption, 2021-07 month-end proxy (exact day not supplied)
    '10373QBA7': ('2021-07-31', 100.125),  # BP Capital Markets America Inc; user-confirmed redemption, 2021-07 month-end proxy (exact day not supplied)
    '774341AH4': ('2021-08-31', 100),  # Rockwell Collins Inc; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '03523TBX5': ('2021-06-30', 100),  # Anheuser-Busch InBev Worldwide Inc; user-confirmed redemption, 2021-06 month-end proxy (exact day not supplied)
    '95040QAE4': ('2021-04-30', 100),  # Welltower Inc; user-confirmed redemption, 2021-04 month-end proxy (exact day not supplied)
    '31428XBM7': ('2021-05-31', 100.15),  # FedEx Corp; user-confirmed redemption, 2021-05 month-end proxy (exact day not supplied)
    '20030NBX8': ('2021-06-30', 100),  # Comcast Corp; user-confirmed redemption, 2021-06 month-end proxy (exact day not supplied)
    '29444UAR7': ('2021-06-30', 107.256),  # Equinix Inc; user-confirmed redemption, 2021-06 month-end proxy (exact day not supplied)
    '44701QBD3': ('2021-06-30', 100),  # Huntsman International LLC; user-confirmed redemption, 2021-06 month-end proxy (exact day not supplied)
    '26441CAR6': ('2021-08-31', 100),  # Duke Energy Corp; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '26969PAA6': ('2021-07-31', 100),  # Eagle Materials Inc; user-confirmed redemption, 2021-07 month-end proxy (exact day not supplied)
    '14149YBF4': ('2021-09-30', 100),  # Cardinal Health Inc; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '278865AU4': ('2021-09-30', 100),  # Ecolab Inc; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '278865AY6': ('2021-09-30', 100),  # Ecolab Inc; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '80007RAB1': ('2021-09-30', 100),  # Sands China Ltd; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '824348BB1': ('2021-10-31', 100),  # Sherwin-Williams Co; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '032654AL9': ('2021-10-31', 100),  # Analog Devices Inc; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '032654AM7': ('2021-10-31', 100.1),  # Analog Devices Inc; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '42824CBF5': ('2021-10-31', 100),  # Hewlett Packard Enterprise Co; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '69362BBC5': ('2021-10-31', 100),  # PSEG Power LLC; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '03349MAC9': ('2021-12-31', 100),  # Andeavor; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '256746AF5': ('2021-12-31', 104.38),  # Dollar Tree Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '75513EBW0': ('2021-08-31', 100),  # Raytheon Technologies Corp; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '78355HKB4': ('2021-08-31', 100),  # Ryder System Inc; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '828807DA2': ('2021-08-31', 100),  # Simon Property Group LP; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '828807DB0': ('2021-08-31', 100),  # Simon Property Group LP; user-confirmed redemption, 2021-08 month-end proxy (exact day not supplied)
    '23355LAC0': ('2021-09-30', 100),  # DXC Technology Co; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '23355LAD8': ('2021-09-30', 100.35),  # DXC Technology Co; user-confirmed redemption, 2021-09 month-end proxy (exact day not supplied)
    '78572XAF8': ('2021-10-31', 100),  # Sabra Health Care LP / Sabra Capital Corp; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '878237AG1': ('2021-10-31', 100),  # Tech Data Corp; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '878237AH9': ('2021-10-31', 100),  # Tech Data Corp; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '06051GFZ7': ('2021-10-31', 100),  # Bank of America Corp; user-confirmed redemption, 2021-10 month-end proxy (exact day not supplied)
    '56585ABA9': ('2021-12-31', 100),  # Marathon Petroleum Corp; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '595112BL6': ('2021-11-30', 100),  # Micron Technology Inc; user-confirmed redemption, 2021-11 month-end proxy (exact day not supplied)
    '883556CG5': ('2021-12-31', 100.50),  # Thermo Fisher Scientific Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '00772BAR2': ('2021-12-31', 100),  # AerCap Ireland Capital Ltd / AerCap Global Aviation Trust; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '06738EAS4': ('2022-01-31', 100),  # Barclays PLC; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '17401QAS0': ('2022-01-31', 100),  # Citizens Bank NA; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '883556BR2': ('2021-12-31', 100),  # Thermo Fisher Scientific Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '883556BX9': ('2021-12-31', 100),  # Thermo Fisher Scientific Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '80281LAF2': ('2022-01-31', 100),  # Santander UK Group Holdings PLC; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '559079AK4': ('2022-01-31', 100),  # Magellan Health Inc; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '22160KAK1': ('2021-12-31', 100),  # Costco Wholesale Corp; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '78355HKD0': ('2022-02-28', 100),  # Ryder System Inc; user-confirmed redemption, 2022-02 month-end proxy (exact day not supplied)
    '928563AB1': ('2022-01-31', 100),  # VMware Inc; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '949746SK8': ('2022-01-31', 100),  # Wells Fargo & Co; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '03350WAA7': ('2022-08-31', 100),  # Andeavor Logistics LP / Tesoro Logistics Finance Corp; user-confirmed redemption, 2022-08 month-end proxy (exact day not supplied)
    '00774MAA3': ('2022-03-31', 100),  # AerCap Ireland Capital DAC / AerCap Global Aviation Trust; user-confirmed redemption, 2022-03 month-end proxy (exact day not supplied)
    '035242AP1': ('2022-02-28', 107.054),  # Anheuser-Busch InBev Finance Inc; user-confirmed redemption, 2022-02 month-end proxy (exact day not supplied)
    '651639AT3': ('2022-01-31', 100),  # Newmont Goldcorp Corp; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '05723KAD2': ('2022-01-31', 100),  # Baker Hughes Co; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '0258M0EG0': ('2022-01-31', 100),  # American Express Credit Corp; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '00287YCS6': ('2022-01-31', 100),  # AbbVie Inc; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '375558BH5': ('2022-02-28', 100),  # Gilead Sciences Inc; user-confirmed redemption, 2022-02 month-end proxy (exact day not supplied)
    '674599CK9': ('2022-01-31', 100),  # Occidental Petroleum Corp; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '25746UCR8': ('2021-12-31', 100.15),  # Dominion Energy Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '67078AAC7': ('2021-12-31', 100),  # nVent Finance Sarl; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '929089AD2': ('2021-11-30', 100),  # Voya Financial Inc; user-confirmed redemption, 2021-11 month-end proxy (exact day not supplied)
    '031162CP3': ('2021-12-31', 100),  # Amgen Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '05565QDG0': ('2021-12-31', 100.20),  # BP Capital Markets PLC; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '05565QDL9': ('2021-12-31', 100),  # BP Capital Markets PLC; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '10373QAW0': ('2021-12-31', 100),  # BP Capital Markets America Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '10373QAY6': ('2021-12-31', 100),  # BP Capital Markets America Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '126650CV0': ('2021-12-31', 100),  # CVS Health Corp; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '15189TAR8': ('2021-12-31', 100.125),  # CenterPoint Energy Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '15189WAK6': ('2021-12-31', 100),  # CenterPoint Energy Resources Corp; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '65339KAW0': ('2021-12-31', 100),  # NextEra Energy Capital Holdings Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '65339KBG4': ('2021-12-31', 100),  # NextEra Energy Capital Holdings Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '65339KBH2': ('2021-12-31', 100),  # NextEra Energy Capital Holdings Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '65339KBQ2': ('2021-12-31', 100),  # NextEra Energy Capital Holdings Inc; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '816851BF5': ('2021-12-31', 100),  # Sempra Energy; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '871829BG1': ('2021-12-31', 100),  # Sysco Corp; user-confirmed redemption, 2021-12 month-end proxy (exact day not supplied)
    '19828JAB4': ('2022-01-31', 100),  # Columbia Property Trust Operating Partnership; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '25389JAQ9': ('2022-02-28', 100),  # Digital Realty Trust LP; user-confirmed redemption, 2022-02 month-end proxy (exact day not supplied)
    '48128BAB7': ('2022-01-31', 100),  # JPMorgan Chase & Co; user-confirmed redemption, 2022-01 month-end proxy (exact day not supplied)
    '81180WAV3': ('2022-02-28', 100),  # Seagate HDD Cayman; user-confirmed redemption, 2022-02 month-end proxy (exact day not supplied)
    '44644AAH0': ('2022-03-31', 100),  # Huntington National Bank; user-confirmed redemption, 2022-03 month-end proxy (exact day not supplied)
    '120568BA7': ('2022-03-31', 100),  # Bunge Ltd Finance Corp; user-confirmed redemption, 2022-03 month-end proxy (exact day not supplied)

}

# Other reviewed bonds not confirmed as early redemptions.
NOT_REDEEMED_EARLY = {
    '25470DAT6': 'Discovery Communications LLC - ineligible per manual audit',
    '92343VDC5': 'Verizon Communications Inc - ineligible per manual audit',
    '91087BAY6': 'United Mexican States - ineligible per manual audit',
    '91087BAZ3': 'United Mexican States - ineligible per manual audit',
    '91087BBA7': 'United Mexican States - ineligible per manual audit',
    '91087BBB5': 'United Mexican States - ineligible per manual audit',
    '91087BBC3': 'United Mexican States - ineligible per manual audit',
    '91087BBD1': 'United Mexican States - ineligible per manual audit',
    '559080AL0': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AM8': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AN6': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AP1': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AQ9': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '035242AM8': 'Anheuser-Busch InBev Finance Inc - ineligible per manual audit',
    '035242AN6': 'Anheuser-Busch InBev Finance Inc - ineligible per manual audit',
    '29336TAE0': 'EnLink Midstream LLC - ineligible per manual audit',
    '20605PAJ0': 'Concho Resources Inc - ineligible per manual audit',
    '84861TAG3': 'Spirit Realty LP - ineligible per manual audit',
    '03349MAB1': 'Andeavor - ineligible per manual audit',
    '92343VDU5': 'Verizon Communications Inc - ineligible per manual audit',
    '92343VDV3': 'Verizon Communications Inc - ineligible per manual audit',
    '26885BAE0': 'EQM Midstream Partners LP - ineligible per manual audit',
    '485170AQ7': 'Kansas City Southern - ineligible per manual audit',
    '485170BC7': 'Kansas City Southern - ineligible per manual audit',
    '84861TAJ7': 'Spirit Realty LP - ineligible per manual audit',
    '264414AX1': 'Duke Realty Limited Partnership - ineligible per manual audit',
    '92340LAC3': 'VEREIT Operating Partnership LP - ineligible per manual audit',
    '09857LAR9': 'Booking Holdings Inc - ineligible per manual audit',
    '161175CM4': 'Charter Communications Operating LLC - ineligible per manual audit',
    '837004CK4': 'Dominion Energy South Carolina Inc - ineligible per manual audit',
    '517834AG2': 'Las Vegas Sands Corp - ineligible per manual audit',
    '78349AAC7': 'RWJ Barnabas Health Inc - ineligible per manual audit',
    '29336TAA8': 'EnLink Midstream LLC - ineligible per manual audit',
    '29336UAG2': 'EnLink Midstream Partners LP - ineligible per manual audit',
    '103730BF6': 'BP Capital Markets America Inc - ineligible per manual audit',
    '53227JAD6': 'Life Storage LP - ineligible per manual audit',
    '31572UAF3': 'Fibria Overseas Finance Ltd - ineligible per manual audit',
    '00217GAA1': 'Aptiv PLC - ineligible per manual audit',
    '20030NDZ1': 'Comcast Corporation - ineligible per manual audit',
    '10373QBF6': 'BP Capital Markets America Inc - ineligible per manual audit',
    '032511BN6': 'Anadarko Petroleum Corporation - ineligible per manual audit',
    '92340LAA7': 'VEREIT Operating Partnership LP - ineligible per manual audit',
    '92340LAD1': 'VEREIT Operating Partnership LP - ineligible per manual audit',
    '92343VDR2': 'Verizon Communications Inc - ineligible per manual audit',
    '92857WBJ8': 'Vodafone Group PLC - ineligible per manual audit',
    '29336UAF4': 'EnLink Midstream Partners LP - ineligible per manual audit',
    '723787AV9': 'Pioneer Natural Resources Company - ineligible per manual audit',
    '92343VDS0': 'Verizon Communications Inc - ineligible per manual audit',
    '71654QCB6': 'Petróleos Mexicanos (PEMEX) - ineligible per manual audit',
    '25470DAD1': 'Discovery Communications LLC - ineligible per manual audit',
    '25470DAG4': 'Discovery Communications LLC - ineligible per manual audit',
    '25470DAJ8': 'Discovery Communications LLC - ineligible per manual audit',
    '92343VBT0': 'Verizon Communications Inc - ineligible per manual audit',
    '92343VCR3': 'Verizon Communications Inc - ineligible per manual audit',
    '125896BM1': 'CMS Energy Corporation - ineligible per manual audit',
    '302154EA3': 'Export-Import Bank of Korea - ineligible per manual audit',
    '302154EB1': 'Export-Import Bank of Korea - ineligible per manual audit',
    '302154EC9': 'Export-Import Bank of Korea - ineligible per manual audit',
    '559080AJ5': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AC0': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AF3': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '559080AG1': 'Magellan Midstream Partners LP - ineligible per manual audit',
    '69352PAH6': 'PPL Capital Funding Inc - ineligible per manual audit',
    '29364WAK4': 'Entergy Louisiana LLC - ineligible per manual audit',
    '337932AC1': 'FirstEnergy Corp - ineligible per manual audit',
    '73755LAK3': 'Potash Corporation of Saskatchewan Inc - ineligible per manual audit',
    '71654QBW1': 'Petróleos Mexicanos (PEMEX) - ineligible per manual audit',
    '92343VAW4': 'Verizon Communications Inc - ineligible per manual audit',
    '92343VBS2': 'Verizon Communications Inc - ineligible per manual audit',
    '92344GAX4': 'Verizon Communications Inc - ineligible per manual audit',
    '00507UAU5': 'Allergan Funding SCS - ineligible per manual audit',
    '774341AF8': 'Rockwell Collins Inc - ineligible per manual audit',
    '29336UAC1': 'EnLink Midstream Partners LP - ineligible per manual audit',
    '29336UAD9': 'EnLink Midstream Partners LP - ineligible per manual audit',
    '008916AN8': 'Agrium Inc - ineligible per manual audit',
    '31410HAQ4': 'Macy\'s Inc - ineligible per manual audit',
    '302154EM7': 'Export-Import Bank of Korea - ineligible per manual audit',
    '92343VAK0': 'Verizon Communications Inc - ineligible per manual audit',
    '92343VCK8': 'Verizon Communications Inc - ineligible per manual audit',
    '92343VCM4': 'Verizon Communications Inc - ineligible per manual audit',
    '92343VCX0': 'Verizon Communications Inc - ineligible per manual audit',
    '931422AK5': 'Walgreen Co - ineligible per manual audit',
    '884903BM6': 'Thomson Reuters Corporation - ineligible per manual audit',
    '69352PAJ2': 'PPL Capital Funding Inc - ineligible per manual audit',
    '302154ED7': 'Export-Import Bank of Korea - ineligible per manual audit',
    '302154EE5': 'Export-Import Bank of Korea - ineligible per manual audit',
    '690732AE2': 'Owens & Minor Inc - ineligible per manual audit',
    '12513GBA6': 'CDW LLC - ineligible per manual audit',
    '302154EG0': 'Export-Import Bank of Korea - ineligible per manual audit',
    '52107QAG0': 'Lazard Group LLC - ineligible per manual audit',
    '125509BQ1': 'Cigna Holding Company - ineligible per manual audit',
    '232820AK6': 'Cytec Industries Inc - ineligible per manual audit',
    '302154EJ4': 'Export-Import Bank of Korea - ineligible per manual audit',
    '302154EK1': 'Export-Import Bank of Korea - ineligible per manual audit',
    '302154EL9': 'Export-Import Bank of Korea - ineligible per manual audit',
    '92343VAP9': 'Verizon Communications Inc - ineligible per manual audit',
    '29082HAB8': 'Embraer Netherlands Finance BV - ineligible per manual audit',
    '636180BM2': 'National Fuel Gas Company - ineligible per manual audit',
    '92343VAF1': 'Verizon Communications Inc - ineligible per manual audit',
    '110122DA3': 'Bristol Myers Squibb Co - ineligible per manual audit',
    '44932HAC7': 'IBM Credit LLC - ineligible per manual audit',
    '25470DAS8': 'Discovery Communications LLC - ineligible per manual audit',
    '78409VBL7': 'S&P Global Inc - ineligible per manual audit',
    '78409VBM5': 'S&P Global Inc - ineligible per manual audit',
    '655044AN5': 'Noble Energy Inc - ineligible per manual audit',
    '20605PAL5': 'Concho Resources Inc - ineligible per manual audit',
    '907818EX2': 'Union Pacific Corp - ineligible per manual audit',
    '74435KAB1': 'Prudential PLC - ineligible per manual audit',
    '487836BV9': 'Kellogg Co - ineligible per manual audit',
    '02209SBA0': 'Altria Group Inc - ineligible per manual audit',
    '11135FAX9': 'Broadcom Inc - ineligible per manual audit',
    '125523AE0': 'Cigna Corp - ineligible per manual audit',
    '125523BR0': 'Cigna Corp - ineligible per manual audit',
    '44932HAH6': 'IBM Credit LLC - ineligible per manual audit',
    '44932HAM5': 'IBM Credit LLC - ineligible per manual audit',
    '23355LAA4': 'DXC Technology Co - ineligible per manual audit',
    '754730AE9': 'Raymond James Financial Inc - ineligible per manual audit',
    '15189TAT4': 'CenterPoint Energy Inc - ineligible per manual audit',
    '939648AE1': 'Washington Prime Group LP - ineligible per manual audit',
    '191216BW9': 'Coca-Cola Co - ineligible per manual audit',
    '191216BZ2': 'Coca-Cola Co - ineligible per manual audit',
    '92343VDQ4': 'Verizon Communications Inc - ineligible per manual audit',
    '233331BB2': 'DTE Energy Co - ineligible per manual audit',
    '10373QAJ9': 'BP Capital Markets America Inc - ineligible per manual audit',
    '26614NAA0': 'DuPont de Nemours Inc - ineligible per manual audit',
    '37045XCW4': 'General Motors Financial Co Inc - ineligible per manual audit',
    '31428XBU9': 'FedEx Corp - ineligible per manual audit',
    '23355LAF3': 'DXC Technology Co - ineligible per manual audit',
    '05531FAX1': 'BB&T Corp - ineligible per manual audit',
    '65339KBF6': 'NextEra Energy Capital Holdings Inc - ineligible per manual audit',
    '91913YAX8': 'Valero Energy Corp - ineligible per manual audit',

}


def excel_date(series):
    """WRDS-exported date cells were stored as Excel serial numbers."""
    numeric = pd.to_numeric(series, errors='coerce')
    from_serial = pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')
    from_text = pd.to_datetime(series.where(numeric.isna()), errors='coerce')
    return from_serial.fillna(from_text)


def read_wrds(path):
    wrds = pd.read_csv(path, dtype=str, keep_default_na=False)
    needed = {'COMPLETE_CUSIP', 'CALLABLE', 'MAKE_WHOLE', 'MATURITY'}
    wrds['cusip'] = wrds.COMPLETE_CUSIP.str.strip().str.upper()
    wrds = wrds.loc[wrds.cusip.ne('')].drop_duplicates('cusip').copy()
    for col in ('MATURITY', 'NEXT_CALL_DATE', 'CALL_DATE'):
        if col in wrds:
            wrds[col] = excel_date(wrds[col])
    return wrds.set_index('cusip')


def read_last_bond_observations(path):
    """Read the 40 MB zipped source in chunks; retain only one row per CUSIP."""
    columns = {'cusip', 'DATE', 'MATURITY', 'PRICE_EOM', 'YIELD'}
    path = Path(path)
    if not path.exists() and Path(str(path) + '.zip').exists():
        path = Path(str(path) + '.zip')
    latest = {}
    sample_end = None
    source_rows = 0
    for chunk in pd.read_csv(path, usecols=lambda col: col in columns, dtype=str, chunksize=200_000, encoding='utf-8-sig', low_memory=False):
        if not {'cusip', 'DATE', 'MATURITY'}.issubset(chunk):
            raise ValueError('AGG source requires cusip, DATE, MATURITY')
        source_rows += len(chunk)
        chunk['cusip'] = chunk.cusip.fillna('').str.strip().str.upper()
        chunk['DATE'] = pd.to_datetime(chunk.DATE, errors='coerce')
        chunk['MATURITY'] = pd.to_datetime(chunk.MATURITY, errors='coerce')
        chunk = chunk.loc[chunk.cusip.ne('') & chunk.DATE.notna()]
        if chunk.empty:
            continue
        newest = chunk.DATE.max()
        sample_end = newest if sample_end is None else max(sample_end, newest)
        # Source is not assumed to be in date order.
        last_in_chunk = chunk.sort_values('DATE').drop_duplicates('cusip', keep='last')
        for rec in last_in_chunk.to_dict('records'):
            cusip = rec['cusip']
            if cusip not in latest or rec['DATE'] > latest[cusip]['DATE']:
                latest[cusip] = rec
    return pd.DataFrame(latest.values()), sample_end.to_period('M'), source_rows


def scan(bonds, wrds, sample_end, start, end, all_history=False):
    bonds = bonds.copy()
    # The source maturity takes priority; WRDS can fill any missing source date.
    joined = bonds.join(wrds, on='cusip', how='left', rsuffix='_wrds')
    joined['maturity_for_scan'] = joined.MATURITY.fillna(joined.MATURITY_wrds)
    joined['last_month'] = joined.DATE.dt.to_period('M')
    joined['first_missing_month'] = joined.last_month + 1
    joined['maturity_month'] = joined.maturity_for_scan.dt.to_period('M')
    # Right-censored sample-end rows and scheduled maturities aren't early exits.
    joined = joined.loc[joined.first_missing_month.le(sample_end) & joined.maturity_month.gt(joined.first_missing_month)].copy()
    history_count = len(joined)
    if not all_history:
        joined = joined.loc[joined.last_month.between(start, end)].copy()

    # Already audited; remove from the list of NEW candidates for review.
    reviewed = set(EARLY_REDEMPTIONS) | set(NOT_REDEEMED_EARLY)
    joined = joined.loc[~joined.cusip.isin(reviewed)].copy()

    joined['months_until_maturity'] = (joined.maturity_month.astype('int64') - joined.first_missing_month.astype('int64'))
    joined['tapestry_like'] = (joined.CALLABLE.eq('Y') & joined.MAKE_WHOLE.eq('Y') & joined.NEXT_CALL_DATE.gt(joined.DATE) 
                                & pd.to_numeric(joined.NEXT_CALL_PRICE, errors='coerce').eq(100))
    joined['review_priority'] = '3 - Other premature exit'
    joined.loc[joined.CALLABLE.eq('Y'), 'review_priority'] = '2 - Callable premature exit'
    joined.loc[joined.tapestry_like, 'review_priority'] = '1 - Tapestry-like call terms'
    joined['maturity_match'] = (joined.MATURITY.dt.to_period('M').eq(joined.MATURITY_wrds.dt.to_period('M')))
    joined['maturity_match'] = joined.maturity_match.astype('boolean')
    joined.loc[joined.MATURITY_wrds.isna() | joined.MATURITY.isna(), 'maturity_match'] = pd.NA

    result = pd.DataFrame({
        'review_priority': joined.review_priority,
        'cusip': joined.cusip,
        'issuer': joined.PROSPECTUS_ISSUER_NAME,
        'last_observed_month': joined.last_month.astype(str),
        'first_missing_month': joined.first_missing_month.astype(str),
        'contractual_maturity': joined.maturity_for_scan.dt.strftime('%Y-%m-%d'),
        'months_until_maturity': joined.months_until_maturity,
        'callable': joined.CALLABLE,
        'make_whole': joined.MAKE_WHOLE,
        'next_scheduled_call_date': joined.NEXT_CALL_DATE.dt.strftime('%Y-%m-%d'),
        'next_scheduled_call_price_NOT_actual': joined.NEXT_CALL_PRICE,
        'wrds_maturity_matches': joined.maturity_match,
        'last_recorded_clean_price': joined.get('PRICE_EOM'),
        'last_recorded_yield': joined.get('YIELD'),
        'verification_status': 'POTENTIAL EXIT - NOT VERIFIED',
    })
    result = result.sort_values(['review_priority', 'last_observed_month', 'cusip'], ascending=[True, True, True]).reset_index(drop=True)
    return result, history_count


# --- DISABLED: terminal-return calculations (not part of bond flagging) ---
# Preserved for future diagnostics; the scanner below does not call these.
# def number(value):
#     """Read cash amounts, allowing source $, commas, and percent signs."""
#     if pd.isna(value):
#         return float('nan')
#     clean = str(value).strip().replace('$', '').replace(',', '').rstrip('%')
#     return pd.to_numeric(clean, errors='coerce')
#
#
# def days_30_360(start, end):
#     """Approximate 30/360 year fraction used for conventional coupon bonds."""
#     return (360 * (end.year - start.year)+ 30 * (end.month - start.month)+ min(end.day, 30) - min(start.day, 30)) / 360.0
#
#
# def coupon_cashflows(row, event_date):
#     """Return coupon cash paid, accrued interest at redemption, estimated AI at t.
#     All monetary amounts per $100 face; COUPON is the annual coupon in %.
#     Requires a conventional fixed-rate coupon calendar. No guessed schedule."""
#     rate = number(row['COUPON'])
#     freq_raw = number(row['NCOUPS'])
#     if not pd.notna(rate) or not pd.notna(freq_raw) or rate < 0:
#         raise ValueError('missing or invalid coupon or frequency')
#     if rate == 0:
#         return 0.0, 0.0, 0.0
#     if freq_raw not in (1, 2, 4, 12):
#         raise ValueError('unsupported coupon frequency')
#     frequency = int(freq_raw)
#     next_coupon = pd.to_datetime(row['nextcoup'], errors='coerce')
#     asof_date = row['DATE']
#     if pd.isna(next_coupon) or not (asof_date < next_coupon):
#         raise ValueError('missing or stale next coupon date')
#
#     # Start from the last scheduled coupon preceding the valuation date.
#     step_months = 12 // frequency
#     previous_coupon = next_coupon - pd.DateOffset(months=step_months)
#     coupons_paid = 0.0
#     due = next_coupon
#     while due <= event_date:
#         coupons_paid += rate / frequency
#         previous_coupon = due
#         due = due + pd.DateOffset(months=step_months)
#
#     ai_at_purchase_estimate = rate * days_30_360(
#         next_coupon - pd.DateOffset(months=step_months), asof_date
#     )
#     ai_at_redemption = rate * days_30_360(previous_coupon, event_date)
#     if ai_at_purchase_estimate < -1e-9 or ai_at_redemption < -1e-9:
#         raise ValueError('invalid accrued-interest dates')
#     return coupons_paid, ai_at_redemption, ai_at_purchase_estimate
#
#
# def compute_terminal_returns(bond_path):
#     """Estimate t -> redemption-month returns for the verified event registry.
#     Uses month t immediately BEFORE the event month, even if the raw dataset
#     contains an observation in the redemption month. Existing RET_EOM is
#     included only for comparison, never overwritten by this scanner."""
#     path = Path(bond_path)
#     if not path.exists() and Path(str(path) + '.zip').exists():
#         path = Path(str(path) + '.zip')
#     needed = {'cusip', 'DATE', 'PRICE_EOM', 'COUPACC', 'COUPON',
#               'NCOUPS', 'nextcoup', 'RET_EOM'}
#     subsets = []
#     for chunk in pd.read_csv(
#         path, usecols=lambda c: c in needed, dtype=str,
#         chunksize=200_000, encoding='utf-8-sig', low_memory=False
#     ):
#         if not needed.issubset(chunk):
#             raise ValueError(f'Bond history missing terminal cashflow columns: {needed - set(chunk)}')
#         chunk['cusip'] = chunk.cusip.fillna('').str.strip().str.upper()
#         small = chunk.loc[chunk.cusip.isin(EARLY_REDEMPTIONS)].copy()
#         if not small.empty:
#             small['DATE'] = pd.to_datetime(small.DATE, errors='coerce')
#             subsets.append(small)
#     history = (pd.concat(subsets, ignore_index=True)
#                if subsets else pd.DataFrame(columns=list(needed)))
#     history['month'] = history.DATE.dt.to_period('M')
#
#     output = []
#     for cusip, (date, principal_payout) in sorted(EARLY_REDEMPTIONS.items()):
#         if len(date) != 10:  # Exact redemption day unknown.
#             continue
#         event_date = pd.Timestamp(date)
#         event_month = event_date.to_period('M')
#         decision_month = event_month - 1
#         own = history.loc[history.cusip.eq(cusip)]
#         prior = own.loc[own.month.eq(decision_month)].sort_values('DATE')
#         actual = own.loc[own.month.eq(event_month)].sort_values('DATE')
#         info = {
#             'cusip': cusip, 'decision_month': str(decision_month),
#             'redemption_month': str(event_month),
#             'R_terminal': float('nan'), 'R_terminal_pct': float('nan'),
#             'purchase_clean_price_per_100': float('nan'),
#             'purchase_accrued_interest_per_100': float('nan'),
#             'redemption_accrued_interest_per_100': float('nan'),
#             'interim_coupons_per_100': float('nan'),
#             'recorded_event_month_RET_EOM': float('nan'),
#             'calculation_status': 'MISSING PRIOR-MONTH DATA',
#         }
#         if not actual.empty:
#             source_return = str(actual.iloc[-1]['RET_EOM']).strip()
#             # The source expresses realized returns with an explicit percent sign.
#             if source_return.endswith('%'):
#                 info['recorded_event_month_RET_EOM'] = (
#                     number(source_return) / 100.0
#                 )
#         if prior.empty:
#             output.append(info)
#             continue
#
#         row = prior.iloc[-1]
#         price = number(row['PRICE_EOM'])
#         if pd.isna(price) or price <= 0:
#             info['calculation_status'] = 'INVALID PRIOR-MONTH PRICE'
#             output.append(info)
#             continue
#         try:
#             paid_coupons, final_ai, estimated_purchase_ai = coupon_cashflows(
#                 row, event_date
#             )
#             purchase_ai = number(row['COUPACC'])
#             if pd.isna(purchase_ai):
#                 purchase_ai = estimated_purchase_ai
#             dirty_purchase = price + purchase_ai
#             if not (pd.notna(dirty_purchase) and dirty_purchase > 0):
#                 raise ValueError('invalid purchase dirty price')
#             if row['DATE'] >= event_date:
#                 raise ValueError('valuation on/after redemption date')
#             r_terminal = (
#                 (principal_payout + final_ai + paid_coupons)
#                 / dirty_purchase - 1.0
#             )
#             info.update({
#                 'R_terminal': r_terminal,
#                 'R_terminal_pct': 100.0 * r_terminal,
#                 'purchase_clean_price_per_100': price,
#                 'purchase_accrued_interest_per_100': purchase_ai,
#                 'redemption_accrued_interest_per_100': final_ai,
#                 'interim_coupons_per_100': paid_coupons,
#                 'calculation_status': 'ESTIMATED - 30/360 INTEREST',
#             })
#         except (ValueError, TypeError, OverflowError) as exc:
#             info['calculation_status'] = f'UNRESOLVED CASH FLOWS: {exc}'
#         output.append(info)
#     return pd.DataFrame(output)
#
# --- END DISABLED terminal-return calculations ---

def main():
    wrds = read_wrds(REFERENCE)
    bonds, sample_end, source_rows = read_last_bond_observations(DEFAULT_BONDS)
    result, all_exits = scan(bonds, wrds, sample_end, pd.Period(START_MONTH, 'M'), pd.Period(END_MONTH, 'M'))

    print('\nEARLY REDEMPTION REVIEW (NOT CONFIRMED REDEMPTIONS)')
    print(f'AGG observations: {source_rows:,} | Unique bonds: {len(bonds):,}')
    print(f'WRDS CUSIPs: {len(wrds):,} | Last source month: {sample_end}')
    print(f'Potential premature exits before prior-review exclusions: {all_exits:,}')
    print(f'Confirmed early redemptions excluded: {len(EARLY_REDEMPTIONS):,}')
    print(f'Other reviewed bonds excluded: {len(NOT_REDEEMED_EARLY):,}')
    print(f'Candidates in review window {START_MONTH} through {END_MONTH}: {len(result):,}')
    if not result.empty:
        print(result.review_priority.value_counts().sort_index().to_string())
        print('\nALL FLAGGED BONDS FOR REVIEW:')
        print(result[['review_priority', 'cusip', 'issuer', 'last_observed_month',
                      'contractual_maturity']].to_string(index=False, max_rows=None))
    else:
        print('No new premature-exit candidates found.')

    print('WARNING: A premature exit can be a redemption OR data/index coverage loss.')
    print('Do not use this report for point-in-time investment eligibility.')


if __name__ == '__main__':
    main()
