-- Verify company funding claims against database
SELECT company_name, round_type, amount_usd, lead_investor, announced_date
FROM funding_rounds
WHERE LOWER(company_name) LIKE LOWER('%{{company_name}}%')
