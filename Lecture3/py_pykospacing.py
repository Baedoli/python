
# Import packages
from pykospacing import Spacing
# Define sentence
sent = '복 있는 사람은 악인들의 꾀를 따르지 아니하며 죄인들의 길에 서지 아니하며 ' \
       '오만한 자들의 자리에 앉지 아니하고 오직 여호와의 율법을 즐거워하여 그의 율법을 주야로 묵상하는도다'
# Create no spaced sentence
new_sent = sent.replace(" ", '')
print(new_sent)
# Apply spacing
spacing = Spacing()
kospacing_sent = spacing(new_sent)
print(sent)
print(kospacing_sent)

