import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LeaveRequest:
    employee: str
    leave_type: str
    start_date: str
    end_date: str
    num_days: int
    status: str
    request_date: str

class LeaveSystem:
    def __init__(self):
        self.employees = {
            "Alice": {"Sick Leave": 5, "Annual Leave": 10, "Maternity Leave": 5},
            "Bob": {"Sick Leave": 8, "Annual Leave": 15, "Maternity Leave": 0},
            "Charlie": {"Sick Leave": 2, "Annual Leave": 12, "Maternity Leave": 0},
        }
        self.leave_history: List[LeaveRequest] = []
        self.leave_types = ["Sick Leave", "Annual Leave", "Maternity Leave"]
        self.setup_llm()

    def setup_llm(self):
        """Initialize LangChain components with improved natural language processing"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        
        response_schemas = [
            ResponseSchema(name="leave_types", description="Array of leave types being queried (can be empty for all types, or contain specific types like ['Sick Leave', 'Annual Leave'])"),
            ResponseSchema(name="num_days", description="Number of days requested (integer or null)"),
            ResponseSchema(name="start_date", description="Start date of leave in YYYY-MM-DD format (or null)"),
            ResponseSchema(name="action", description="The action being requested (request/cancel/check/view)")
        ]

        self.parser = StructuredOutputParser.from_response_schemas(response_schemas)


        self.extract_template = """
        Extract leave-related information from the user's input. Be flexible in understanding different ways users might express their intentions.
        
        Valid leave types are: Sick Leave, Annual Leave, Maternity Leave
        Valid actions are: request, cancel, check, view
        
        For the leave_types field:
        - Return an empty array [] if the user wants to check all leave types
        - Return an array with specific leave types if mentioned
        
        Examples:
        Input: "I need to take 3 days off next week for medical reasons"
        Output: {{"leave_types": ["Sick Leave"], "num_days": 3, "start_date": null, "action": "request"}}
        
        Input: "I want to check my leave balance"
        Output: {{"leave_types": [], "num_days": null, "start_date": null, "action": "check"}}
        
        Input: "How many sick days do I have left?"
        Output: {{"leave_types": ["Sick Leave"], "num_days": null, "start_date": null, "action": "check"}}
        
        Input: "Show my vacation and sick leave balance"
        Output: {{"leave_types": ["Annual Leave", "Sick Leave"], "num_days": null, "start_date": null, "action": "check"}}
        
        Input: "Cancel my leave"
        Output: {{"leave_types": [], "num_days": null, "start_date": null, "action": "cancel"}}
        
        Input: "Show my leave history"
        Output: {{"leave_types": [], "num_days": null, "start_date": null, "action": "view"}}
        
        Current user input: {input}
        
        {format_instructions}
        """

        self.extract_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_template(self.extract_template)
        )

    def get_date_input(self, prompt: str) -> str:
        """Get and validate date input from user"""
        while True:
            date_str = input(prompt)
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-02-01)")

    def get_leave_type_input(self) -> str:
        """Get leave type selection from user"""
        while True:
            print("\nAvailable leave types:")
            for i, leave_type in enumerate(self.leave_types, 1):
                print(f"{i}. {leave_type}")
            try:
                choice = int(input("Select leave type (1-3): "))
                if 1 <= choice <= len(self.leave_types):
                    return self.leave_types[choice - 1]
            except ValueError:
                pass
            print("Invalid choice. Please select a number between 1 and 3.")

    def get_number_of_days(self, max_days: int) -> int:
        """Get and validate number of leave days"""
        while True:
            try:
                days = int(input(f"Enter number of days (max {max_days}): "))
                if 1 <= days <= max_days:
                    return days
                print(f"Please enter a number between 1 and {max_days}")
            except ValueError:
                print("Please enter a valid number")

    def check_balance(self, employee: str, leave_types: List[str] = None) -> str:
        """
        Check leave balance for an employee
        Args:
            employee: Employee name
            leave_types: List of specific leave types to check. If None or empty, check all types.
        """
        if employee not in self.employees:
            return f"Employee {employee} not found."
        
        balance = self.employees[employee]
        
        # If no specific leave types requested, show all
        if not leave_types:
            return "\n".join([f"{leave_type}: {days} days" for leave_type, days in balance.items()])
        
        # Show only requested leave types
        requested_balance = {lt: balance[lt] for lt in leave_types if lt in balance}
        if not requested_balance:
            return "No valid leave types specified."
        
        return "\n".join([f"{leave_type}: {days} days" for leave_type, days in requested_balance.items()])

    def request_leave(self, employee: str, leave_type: str, start_date: str, num_days: int) -> str:
        """Process a leave request"""
        if employee not in self.employees:
            return f"Employee {employee} not found."
        
        if leave_type not in self.employees[employee]:
            return f"Invalid leave type: {leave_type}"
        
        current_balance = self.employees[employee][leave_type]
        if current_balance < num_days:
            return f"Insufficient {leave_type} balance. You have {current_balance} days available."
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = start + timedelta(days=num_days - 1)
            end_date = end.strftime("%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD format."
        
        leave_request = LeaveRequest(
            employee=employee,
            leave_type=leave_type,
            start_date=start_date,
            end_date=end_date,
            num_days=num_days,
            status="approved",
            request_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        self.employees[employee][leave_type] -= num_days
        self.leave_history.append(leave_request)
        
        return f"Leave request approved. {num_days} days of {leave_type} scheduled from {start_date} to {end_date}."

    def handle_cancel_leave(self, employee: str) -> str:
        """Handle leave cancellation process"""
        active_leaves = [leave for leave in self.leave_history 
                        if leave.employee == employee and leave.status == "approved"]
        
        if not active_leaves:
            return "No active leave requests found."
        
        print("\nActive leave requests:")
        for i, leave in enumerate(active_leaves, 1):
            print(f"{i}. {leave.leave_type}: {leave.num_days} days from {leave.start_date} to {leave.end_date}")
        
        while True:
            try:
                choice = int(input("\nSelect the leave to cancel (or 0 to go back): "))
                if choice == 0:
                    return "Leave cancellation cancelled."
                
                if 1 <= choice <= len(active_leaves):
                    leave_to_cancel = active_leaves[choice - 1]
                    self.employees[employee][leave_to_cancel.leave_type] += leave_to_cancel.num_days
                    leave_to_cancel.status = "cancelled"
                    
                    cancellation_record = LeaveRequest(
                        employee=employee,
                        leave_type=leave_to_cancel.leave_type,
                        start_date=leave_to_cancel.start_date,
                        end_date=leave_to_cancel.end_date,
                        num_days=leave_to_cancel.num_days,
                        status="cancelled",
                        request_date=datetime.now().strftime("%Y-%m-%d")
                    )
                    self.leave_history.append(cancellation_record)
                    
                    return (f"Cancelled {leave_to_cancel.num_days} days of {leave_to_cancel.leave_type} "
                            f"from {leave_to_cancel.start_date}. Leave balance updated.")
                
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def view_history(self, employee: str) -> str:
        """View leave history for an employee"""
        employee_leaves = [leave for leave in self.leave_history if leave.employee == employee]
        
        if not employee_leaves:
            return "No leave history found."
        
        history_str = "\nLeave History:\n"
        for leave in employee_leaves:
            history_str += f"- {leave.leave_type}: {leave.num_days} days from {leave.start_date} to {leave.end_date} ({leave.status})\n"
        
        return history_str

    async def process_input(self, user_input: str, current_employee: str) -> str:
        """Process user input using improved natural language understanding"""
        try:
            extraction_result = await self.extract_chain.ainvoke({
                "input": user_input,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            leave_info = self.parser.parse(extraction_result["text"])

            if leave_info["action"] == "check":
                return self.check_balance(current_employee, leave_info["leave_types"])

            elif leave_info["action"] == "request":
                # If multiple leave types specified for request, use the first one
                leave_type = leave_info["leave_types"][0] if leave_info["leave_types"] else self.get_leave_type_input()

                num_days = leave_info["num_days"]
                if not num_days:
                    current_balance = self.employees[current_employee][leave_type]
                    print(f"\nYou have {current_balance} days of {leave_type} available.")
                    num_days = self.get_number_of_days(current_balance)

                start_date = leave_info["start_date"]
                if not start_date:
                    start_date = self.get_date_input("\nEnter start date (YYYY-MM-DD): ")

                return self.request_leave(
                    current_employee,
                    leave_type,
                    start_date,
                    num_days
                )

            elif leave_info["action"] == "cancel":
                return self.handle_cancel_leave(current_employee)

            elif leave_info["action"] == "view":
                return self.view_history(current_employee)

            return "I'm sorry, I didn't understand that request. Please try again."

        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"

async def main():
    """Main application loop"""
    try:
        leave_system = LeaveSystem()
        print("\nWelcome to the Leave Management System!")
    
        
        while True:
            print("\nPlease log in (Available employees: Alice, Bob, Charlie)")
            current_employee = input("Enter your name (or 'exit' to quit): ").strip()
            
            if current_employee.lower() == 'exit':
                break
                
            if current_employee not in leave_system.employees:
                print("Employee not found. Please try again.")
                continue
            
            print(f"\nLogged in as: {current_employee}")
            print("\nYou can:")
            print("- Check your leave balance (all or specific types)")
            print("- Request a leave")
            print("- Cancel a leave")
            print("- View your leave history")
        
            
            while True:
                print("\nWhat would you like to do? (type 'logout' to switch user or 'exit' to quit)")
                user_input = input("> ")
                
                if user_input.lower() == 'exit':
                    return
                if user_input.lower() == 'logout':
                    break
                
                response = await leave_system.process_input(user_input, current_employee)
                print("\nResponse:", response)
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())