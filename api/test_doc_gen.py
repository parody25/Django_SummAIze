from helpers import replace_text_in_doc, flatten_dict, generate_doc_response
from helpers import create_credit_application, get_credit_application, update_credit_application_section
from data_models.credit_application import CreditApplication

# create_credit_application("123")

response_data = {
  "borrower_name" : "",
  "date_of_application" : "26th March, 2025",
  "type_of_business" : "",
  "risk_rating" : "",
  "new_or_existing" : "Existing",
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

update_credit_application_section("123", "date", "March 22nd")
update_credit_application_section("123", "credit_application_name", 'Emaar')

# credit_application = get_credit_application('a27f23f5-4718-4377-b7c0-338913a8a086')

# print(flatten_dict(credit_application.model_dump()))


generate_doc_response("123")