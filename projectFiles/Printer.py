class Printer:

	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

	def print_title(title):
		print(Printer.HEADER + Printer.BOLD + title + Printer.ENDC)

	def print_line(line):
		print(Printer.OKBLUE + line + Printer.ENDC)

	def print_empty_lines(n):
		for i in range(n):
			print()
	
	def print_separator():
		print(Printer.OKBLUE + "-----------------------------------------------------------------------------" + Printer.ENDC)
		Printer.print_empty_lines(2)
