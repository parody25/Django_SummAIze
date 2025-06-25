from data_models.credit_application import CreditApplication
from data_models.ownership import Owner, Ownership
from data_models.borrower_details import BorrowerDetails

credit_application = CreditApplication()

for field, value in credit_application.model_dump().items():
    print(f"{field}: {value}")
    


credit_application.ownership = Ownership(owners=[
    Owner(name="John Doe", position="CEO", DOB="1980-01-01", percentage_ownership="50.0", net_worth="1000000"),
    Owner(name="Jane Smith", position="CFO", DOB="1985-05-15", percentage_ownership="50.0", net_worth="800000"),
])

ownership_json = credit_application.ownership.model_dump()
# print(ownership_json)


response_data = {
  "borrower_name" : "",
  "date_of_application" : "26th March, 2025",
  "type_of_business" : "",
  "risk_rating" : "",
  "new_or_exising" : "Existing",
  "naics_code" : "",
  "borrower_address" : "",
  "a_telephone" : "",
  "email_address" : "a@b.com",
  "fax_number" : "",
  "branch_number" : "",
  "account_number" : ""
}


borrower_details = BorrowerDetails.model_validate(response_data)
# print(borrower_details.model_dump())



def flatten_dict(data, parent_key='', sep='__'):
    """Recursively flatten a nested dictionary into key-value pairs."""
    flattened = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flattened.update(flatten_dict(item, f"{new_key}{sep}{i}", sep))
                else:
                    flattened[f"{new_key}{sep}{i}"] = item
        else:
            flattened[new_key] = value
    return flattened

def flatten_dict1(d, parent_key=""):
    """Flattens a nested dictionary but keeps only the last-level key names."""
    flat_dict = {}

    def recurse(sub_d):
        if isinstance(sub_d, dict):
            for key, value in sub_d.items():
                if isinstance(value, dict):
                    recurse(value)  # Keep going deeper
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            recurse(item)  # Flatten each dict in list
                        else:
                            flat_dict[key] = value  # Keep non-dict lists as-is
                else:
                    flat_dict[key] = value  # Store final value

    recurse(d)
    return flat_dict



template_data = flatten_dict(credit_application.model_dump())

template_data1 = flatten_dict1(credit_application.model_dump())


print(len(template_data))

print(template_data)

print(template_data1)







