select * from {{ source('csv_sources', 'global_cases') }}