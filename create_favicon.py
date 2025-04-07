import base64
import os
from PIL import Image
import io

# Create the directory if it doesn't exist
os.makedirs('/workspace/docs/source/_static/images', exist_ok=True)

# This is a simple 32x32 favicon in base64 PNG format with KAN colors
favicon_base64 = """
iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAB/UlEQVR42u2Xv2sUQRTHPztnThTB
RgRFBBsREcHCQiwsBAtLQVIJVjaCP8BGCwtBEEHBJqWFlUVKCwsLC0EQQZAUIgpGDUaJCYbzu3zC
eLy92Vnu9m4XH7zszt7MfN6b9968L5xzxR7a5PYLsF8Bdis8PWeeAVVgqJ+xZxrQNHAcmPG1gDXg
C7AM/NxJgD+AJ9oSY40AY8C4fj8DvgB/twswDuyrwKhhnlYCJoEL+v0a+JQnQAtwU7PHFeCccg54
BFwDXgGfgVZWgDuq67e6YaNNKI8DIyL1XGWpWQAGDXNGtAAXgVPAQ+BXCGBIAA4CZ0UAa6sCt4AF
4GHWGRgFrgB3tUJCWiOXjQFVVoC8bFaJPVVZ2A0DHAL2tKlrVLbO6XeEpkwkuwlwQT0fgSgVcAI4
kqGX4n3zRa61C3gD3AQmRKIa8HJLQNVV/23f+wQqPeqHHwPRSQEekXuN9B5I9X5QmrfT5/mKqYq0
O33qh29A415J+5sJuKX/XlfvN/kF70b/mP2Jf9zvbv7+FXhnaCvI+6W0vxbwLWCYFMsgMC/Xe3cH
lPMgeYDyEViS+0MN4G0JQ+TMzZRKsXsXxYZCOeWFFFuEF9KWxf56FrFZiEYDXNDNwmvlFvDd0PaH
cu9UDcZ2/CyhrzfVnv6OfgPsAPr+a1gAvAeu9mu+BWCxR9//A0q3XJ0Knpi3AAAAAElFTkSuQmCC
"""

# Decode the base64 string and open as Image
favicon_data = base64.b64decode(favicon_base64.strip())
favicon = Image.open(io.BytesIO(favicon_data))

# Save the favicon as PNG
favicon.save('/workspace/docs/source/_static/images/favicon.png')

# Save a resized version for different favicon resolutions
favicon_16 = favicon.resize((16, 16), Image.LANCZOS)
favicon_32 = favicon.resize((32, 32), Image.LANCZOS)
favicon_48 = favicon.resize((48, 48), Image.LANCZOS)
favicon_64 = favicon.resize((64, 64), Image.LANCZOS)

# Save these versions
favicon_16.save('/workspace/docs/source/_static/images/favicon-16x16.png')
favicon_32.save('/workspace/docs/source/_static/images/favicon-32x32.png')
favicon_48.save('/workspace/docs/source/_static/images/favicon-48x48.png')
favicon_64.save('/workspace/docs/source/_static/images/favicon-64x64.png')

print("Favicon files created successfully!")