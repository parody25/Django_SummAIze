from data_models.credit_application import CreditApplication
from helpers import create_credit_application, get_credit_application, update_credit_application_section

create_credit_application("123")

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

update_credit_application_section("123", "borrower_details", response_data)

update_credit_application_section("123", "management_analysis", {
    "board_of_directors_profile" : "Hello!"
})

update_credit_application_section("123", "security", {
    "real_estate_security" : "Call the cops!"
})

print(get_credit_application("123"))

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

template_data = flatten_dict1(get_credit_application("123").model_dump())
print(template_data)