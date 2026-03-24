"""
MoodTune Project Manager V4.2
Production Utility
"""
import os
import sys
import time

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print(r"""
  __  __                 _ _______                  
 |  \/  |               | |__   __|                 
 | \  / | ___   ___   __| |  | |_   _ _ __   ___    
 | |\/| |/ _ \ / _ \ / _` |  | | | | | '_ \ / _ \   
 | |  | | (_) | (_) | (_| |  | | |_| | | | |  __/   
 |_|  |_|\___/ \___/ \__,_|  |_|\__,_|_| |_|\___|   
                                                    
    Production Manager (V.LSV)
    """)
    print(f"{Colors.ENDC}")

def check_data():
    if not os.path.exists("data/models/tracks_clustered.parquet"):
        print(f"{Colors.FAIL}❌ Critical Error: Data not found.{Colors.ENDC}")
        print("   You must run option [3] 'Prepare Data' first.")
        return False
    return True

def main():
    while True:
        clear_screen()
        print_banner()
        
        print(f"{Colors.CYAN}Select an option:{Colors.ENDC}")
        print(f"  {Colors.BOLD}[1]{Colors.ENDC} 🌐 Launch Web Interface (Browser)")
        print(f"  {Colors.BOLD}[2]{Colors.ENDC} 💻 Launch Terminal App (CLI)")
        print(f"  {Colors.BOLD}[3]{Colors.ENDC} 📊 Prepare Data (Reset DB)")
        print(f"  {Colors.BOLD}[q]{Colors.ENDC} 🚪 Quit")
        
        choice = input(f"\n{Colors.BLUE}➤ Command: {Colors.ENDC}").lower().strip()
        
        if choice == '1':
            if check_data():
                print(f"\n{Colors.GREEN}🚀 Starting Web Server on Port 5000...{Colors.ENDC}")
                os.system(f"{sys.executable} app.py")
                input("\nServer stopped. Press Enter to return...")
        
        elif choice == '2':
            if check_data():
                print(f"\n{Colors.GREEN}🚀 Starting CLI Tool...{Colors.ENDC}")
                os.system(f"{sys.executable} moodtune_recommender.py")
        
        elif choice == '3':
            print(f"\n{Colors.WARNING}⚙️  Starting Data Preparation Process...{Colors.ENDC}")
            os.system(f"{sys.executable} data_preparation.py")
            input(f"\n{Colors.GREEN}Done! Press Enter to return...{Colors.ENDC}")
            
        elif choice == 'q':
            print("Goodbye!")
            sys.exit()

if __name__ == "__main__":
    main()