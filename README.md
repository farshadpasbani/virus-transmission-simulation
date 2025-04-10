# Virus Transmission Simulation

An interactive simulation of virus transmission across multiple communities using the SIR (Susceptible-Infected-Recovered) model.

## Features

- Multi-community virus transmission simulation
- Interactive parameters adjustment
- Real-time visualization of infection spread
- Detailed statistics for each community

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```
git clone https://github.com/YOUR_USERNAME/virus-transmission-simulation.git
cd virus-transmission-simulation
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

## Parameters

- **Number of Communities**: Number of communities in the simulation (2-5)
- **Population per Community**: Number of individuals in each community
- **Infection Rate (β)**: Rate at which susceptible individuals become infected
- **Recovery Rate (γ)**: Rate at which infected individuals recover
- **Travel Probability**: Probability of travel between communities
- **Initial Infected**: Number of initially infected individuals in each community
- **Simulation Days**: Duration of the simulation in days

## License

This project is licensed under the MIT License - see the LICENSE file for details. 