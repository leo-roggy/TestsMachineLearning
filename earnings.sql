SELECT pe.earnings_amount, pe.earnings_is_current, pe.earnings_frequency, pu.user_first_name, pu.user_last_name, pu.user_birth_date
FROM pr_earnings as pe
INNER JOIN pr_user as pu
	ON pr_user_id = pu.id
WHERE user_last_name is not null
	and pe.earnings_type = 'SALARY';