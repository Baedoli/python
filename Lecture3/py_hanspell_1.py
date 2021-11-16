
# Import packages
from hanspell import spell_checker

# Define sentence
sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "

# Apply spell checker
spelled_sent = spell_checker.check(sent)
hanspell_sent = spelled_sent.checked

# Print result
print(hanspell_sent)

