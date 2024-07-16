import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog, messagebox, Text
 
 
def load_data(file_path):
   return pd.read_csv(file_path)
 
 
def plot_well_logs(data):
   fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 10), sharey=True)
 
 
   sns.lineplot(x='GR', y='Depth', data=data, ax=ax[0], color='g')
   ax[0].set_title('Gamma Ray (GR)')
   ax[0].invert_yaxis()
   ax[0].set_ylabel('Depth (ft)')
   ax[0].set_xlabel('GR (API)')
 
 
   sns.lineplot(x='Resistivity', y='Depth', data=data, ax=ax[1], color='b')
   ax[1].set_title('Resistivity')
   ax[1].set_xlabel('Resistivity (ohm.m)')
 
 
   sns.lineplot(x='Porosity', y='Depth', data=data, ax=ax[2], color='r')
   ax[2].set_title('Porosity')
   ax[2].set_xlabel('Porosity (fraction)')
 
 
   sns.lineplot(x='Saturation', y='Depth', data=data, ax=ax[3], color='m')
   ax[3].set_title('Saturation')
   ax[3].set_xlabel('Saturation (fraction)')
 
 
   sns.lineplot(x='Permeability', y='Depth', data=data, ax=ax[4], color='c')
   ax[4].set_title('Permeability')
   ax[4].set_xlabel('Permeability (mD)')
 
 
   plt.tight_layout()
   plt.show()
 
 
def evaluate_reservoir(data):
   conditions = [
       (data['Porosity'] >= 0.15) & (data['Saturation'] >= 0.75) & (data['Permeability'] >= 100),
       (data['Porosity'] < 0.15) | (data['Saturation'] < 0.75) | (data['Permeability'] < 100)
   ]
   choices = ['Good', 'Bad']
   data['Reservoir_Quality'] = np.select(conditions, choices, default='Bad')
   return data
 
 
def train_model(data):
   X = data[['GR', 'Resistivity', 'Porosity', 'Saturation', 'Permeability']]
   y = data['Oil_Production']
 
 
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 
   model = LinearRegression()
   model.fit(X_train, y_train)
 
 
   y_pred = model.predict(X_test)
 
 
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
 
 
   return model, X_test, y_test, y_pred, mse, r2
 
 
def determine_exploitation(data, y_test, y_pred):
   avg_pred_production = np.mean(y_pred)
   avg_actual_production = np.mean(y_test)
   reservoir_quality = data['Reservoir_Quality'].value_counts().idxmax()
 
 
   if avg_pred_production > avg_actual_production * 0.8 and reservoir_quality == 'Good':
       return "This well is good for exploitation."
   else:
       return "This well is not ideal for exploitation."
 
 
def load_and_analyze():
   file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
   if not file_path:
       return
 
 
   try:
       well_log_data = load_data(file_path)
       well_log_data = evaluate_reservoir(well_log_data)
       messagebox.showinfo("Success", "Data loaded and analyzed successfully.")
       plot_well_logs(well_log_data)
       model, X_test, y_test, y_pred, mse, r2 = train_model(well_log_data)
       messagebox.showinfo("Success", "Model trained successfully. Check console for performance metrics.")
 
 
       result = determine_exploitation(well_log_data, y_test, y_pred)
       messagebox.showinfo("Result", result)
 
 
       # Display results in the text box
       results_text.delete("1.0", tk.END)
       results_text.insert(tk.END, f"Mean Squared Error: {mse}\n")
       results_text.insert(tk.END, f"R-squared: {r2}\n")
       results_text.insert(tk.END, f"Reservoir Quality: {well_log_data['Reservoir_Quality'].value_counts().idxmax()}\n")
       results_text.insert(tk.END, f"Exploitation Recommendation: {result}\n")
 
 
       # Rearrange columns to have Reservoir_Quality next to Oil_Production
       cols = list(well_log_data.columns)
       cols.insert(cols.index('Oil_Production') + 1, cols.pop(cols.index('Reservoir_Quality')))
       well_log_data = well_log_data[cols]
 
 
       # Display the DataFrame in the text box
       data_text.delete("1.0", tk.END)
       data_text.insert(tk.END, well_log_data.to_string())
   except Exception as e:
       messagebox.showerror("Error", f"An error occurred: {e}")
 
 
def create_gui():
   global results_text, data_text
 
 
   root = tk.Tk()
   root.title("Well Log Data Analyzer and Reservoir Quality Predictor - The Pycodes")
   root.geometry("1000x700")
 
 
   canvas = tk.Canvas(root, height=600, width=800)
   canvas.pack()
 
 
   frame = tk.Frame(root, bg='#80c1ff', bd=5)
   frame.place(relx=0.5, rely=0.1, relwidth=0.8, relheight=0.2, anchor='n')
 
 
   button = tk.Button(frame, text="Load and Analyze Well Log Data", padx=10, pady=5, fg="white", bg="#263D42",
                      command=load_and_analyze)
   button.pack()
 
 
   results_frame = tk.Frame(root, bg='#80c1ff', bd=5)
   results_frame.place(relx=0.5, rely=0.35, relwidth=0.8, relheight=0.25, anchor='n')
 
 
   results_text = Text(results_frame, wrap=tk.WORD)
   results_text.pack(expand=True, fill='both')
 
 
   data_frame = tk.Frame(root, bg='#80c1ff', bd=5)
   data_frame.place(relx=0.5, rely=0.65, relwidth=0.8, relheight=0.3, anchor='n')
 
 
   data_text = Text(data_frame, wrap=tk.WORD)
   data_text.pack(expand=True, fill='both')
 
 
   root.mainloop()
 
 
if __name__ == "__main__":
   create_gui()
