with us_cases as (
    select * from {{ ref("stg_us_cases") }}
),
use_cases_by as (
    select
        ID as case_id,
        "Filing Year" as filing_year,
        "Case Name" as case_name,
        "Description" as case_description,
        "Case Categories" as case_categories,
        "Principal Laws" as principal_laws,
        string_split("Case Categories", '|') as case_category_list,
    FROM
        us_cases
)

select * from use_cases_by
