with us_cases as (
    SELECT * from {{ ref("stg_us_cases") }}
),
use_cases_1 as (
    SELECT 
        *,
        string_split("Case Categories", '|') as case_categories
    FROM 
        us_cases
)

select * from use_cases_1