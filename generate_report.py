from fpdf import FPDF
import os

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)

# Directory with overlay images
overlay_dir = "overlays"
overlay_files = sorted([f for f in os.listdir(overlay_dir) if f.endswith(".png")])

print(f"Generating PDF for {len(overlay_files)} overlay images...")

# Add each overlay to a page with clean numbering
for idx, filename in enumerate(overlay_files, start=1):
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Test Sample #{idx}", ln=True, align='C')  # Page title
    image_path = os.path.join(overlay_dir, filename)
    pdf.image(image_path, x=25, y=30, w=160)  # You can adjust x/y/w if needed

# Save the final PDF
pdf.output("output_images.pdf")
print("PDF created: output_images.pdf")
