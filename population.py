#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vehicle Ontology Population Tool

This script loads a CSV file containing vehicle data and populates an ontology with vehicle instances and their properties
"""

import csv
import os
import sys
import time
import re
from owlready2 import *
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File constants
ONTOLOGY_FILENAME = "vehicle_ontology.rdf"
OUTPUT_FILENAME = "vehicle_ontology_populated.rdf"
DATA_FILENAME = "data.csv"
ARTIFACTS_DIR = "artifacts"

# Processing constants
MAX_VEHICLES = 1000
BATCH_SIZE = 100

# Classification thresholds
PREMIUM_THRESHOLD = -10000
STANDARD_THRESHOLD = -1000

# Mapping dictionaries for categorization
FUEL_TYPE_MAP = {
    "CNG": "CNG",
    "Diesel": "Diesel",
    "Electricity": "Electricity",
    "Gasoline or E85": "E85",
    "Premium": "Premium",
    "Regular": "Regular",
    "Midgrade": "Midgrade",
    "Premium Gas or Electricity": "Premium",  # Will set hasElectricity separately
    "Premium and Electricity": "Premium",  # Will set hasElectricity separately
    "Regular Gas and Electricity": "Regular",  # Will set hasElectricity separately
    "Regular Gas or Electricity": "Regular",  # Will set hasElectricity separately
}

DRIVE_TYPE_MAP = {
    "Front-Wheel Drive": "FrontWheelDrive",
    "Rear-Wheel Drive": "RearWheelDrive",
    "All-Wheel Drive": "AllWheelDrive",
    "4-Wheel Drive": "FourWheelDrive",
    "4-Wheel or All-Wheel Drive": "FourWheelDrive",
    "Part-time 4-Wheel Drive": "FourWheelDrive",
    "2-Wheel Drive": "FrontWheelDrive",
}

SIZE_CLASS_MAP = {
    "Compact Cars": "CompactSize",
    "Subcompact Cars": "SubcompactSize",
    "Midsize Cars": "MidsizeSize",
    "Large Cars": "LargeSize",
    "Minicompact Cars": "MinicompactSize",
    "Small Pickup Trucks": "PickupSize",
    "Sport Utility Vehicle - 4WD": "SUVSize",
    "Sport Utility Vehicle - 2WD": "SUVSize",
    "Small Sport Utility Vehicle 4WD": "SUVSize",
    "Small Sport Utility Vehicle 2WD": "SUVSize",
    "Standard Sport Utility Vehicle 4WD": "SUVSize",
    "Standard Sport Utility Vehicle 2WD": "SUVSize",
    "Vans, Passenger Type": "VanSize",
}

BODY_STYLE_MAP = {
    "Compact Cars": "Sedan",
    "Subcompact Cars": "Sedan",
    "Midsize Cars": "Sedan",
    "Large Cars": "Sedan",
    "Minicompact Cars": "Coupe",
    "Two Seaters": "Coupe",
    "Small Station Wagons": "Wagon",
    "Midsize Station Wagons": "Wagon",
    "Sport Utility Vehicle - 4WD": "SUV",
    "Sport Utility Vehicle - 2WD": "SUV",
    "Small Sport Utility Vehicle 4WD": "SUV",
    "Small Sport Utility Vehicle 2WD": "SUV",
    "Standard Sport Utility Vehicle 4WD": "SUV",
    "Standard Sport Utility Vehicle 2WD": "SUV",
    "Small Pickup Trucks": "Truck",
    "Small Pickup Trucks 2WD": "Truck",
    "Small Pickup Trucks 4WD": "Truck",
    "Standard Pickup Trucks": "Truck",
    "Standard Pickup Trucks 2WD": "Truck",
    "Standard Pickup Trucks 4WD": "Truck",
    "Standard Pickup Trucks/2wd": "Truck",
    "Vans": "Van",
    "Vans Passenger": "Van",
    "Vans, Cargo Type": "Van",
    "Vans, Passenger Type": "Van",
    "Minivan - 2WD": "Van",
    "Minivan - 4WD": "Van",
}

BOOST_SYSTEM_MAP = {"T": "Turbocharger", "S": "Supercharger"}

# Electric-related fuel types for determining electricity property
ELECTRIC_FUEL_TYPES = [
    "Electricity",
    "Premium Gas or Electricity",
    "Premium and Electricity",
    "Regular Gas and Electricity",
    "Regular Gas or Electricity",
]


def log(message):
    """
    Log messages with timestamp for tracking program execution

    IMPORTANT: Provides consistent logging format throughout the application

    Args:
        message: String message to be logged with timestamp prefix
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")


def ensure_dir(directory):
    """
    Create directory if it doesn't exist

    Args:
        directory: Path to directory that should exist

    Returns:
        None but creates the directory if missing
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        log(f"Created directory: {directory}")


def save_unique_values(values, filename):
    """
    Save unique values to a file for analysis

    Exports sorted list of unique column values to text file
    filtering out None and NaN values for clean output

    Args:
        values: Array of values to save
        filename: Output file path
    """
    with open(filename, "w", encoding="utf-8") as f:
        for value in sorted(values):
            if value is not None and not pd.isna(value):
                f.write(f"{value}\n")
    log(f"Saved {len(values)} unique values to {filename}")


def create_valid_id(text):
    """
    Create a valid ontology ID from input text

    IMPORTANT: Ensures IDs conform to ontology naming requirements
    by replacing special characters and ensuring alphabetic first char

    Args:
        text: Input string to convert to valid ID

    Returns:
        String with special characters replaced by underscores
    """
    if not text:
        return "Unknown"

    valid_id = re.sub(r"[^a-zA-Z0-9]", "_", str(text))  # Replace special chars with underscore

    if not valid_id[0].isalpha():
        valid_id = "V_" + valid_id  # Prefix non-alpha starting IDs
    return valid_id


def safe_numeric_conversion(value):
    """
    Safely convert a value to numeric type, handling edge cases

    Uses Decimal for precision, then converts to float for compatibility
    with careful error handling to prevent runtime exceptions

    Args:
        value: Input value to convert to numeric

    Returns:
        Float value if conversion successful, None otherwise
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(Decimal(str(value)))  # Precision conversion via Decimal
    except (ValueError, TypeError, InvalidOperation):
        log(f"WARNING: Could not convert value '{value}' to a numeric type")
        return None


def get_market_segment(savings):
    """
    Determine market segment based on vehicle savings value

    Uses thresholds to classify vehicles into market segments:
    - Premium: Significant savings (below PREMIUM_THRESHOLD)
    - Standard: Moderate savings (between thresholds)
    - Economy: No savings or costs more (above STANDARD_THRESHOLD)

    Args:
        savings: Numeric value representing consumer savings/spending

    Returns:
        Market segment object from the ontology
    """
    if pd.isna(savings):
        return "StandardMarket"

    if savings < PREMIUM_THRESHOLD:
        return "PremiumMarket"
    elif savings < STANDARD_THRESHOLD:
        return "StandardMarket"
    else:
        return "EconomyMarket"


def populate_ontology():
    """
    Main function to populate the vehicle ontology with data

    IMPORTANT: Orchestrates the entire ontology population workflow:
    - Loads the base ontology and data file
    - Creates mapping dictionaries for classification
    - Processes vehicle records and adds them to the ontology
    - Creates appropriate relationships between individuals
    - Saves the populated ontology

    Returns:
        None but populates and saves the ontology file
    """
    log("Starting ontology population process")

    log("Loading ontology...")
    onto_path.append(".")

    try:
        world = World()
        onto = world.get_ontology(f"file://{ONTOLOGY_FILENAME}").load()
        log(f"Ontology '{ONTOLOGY_FILENAME}' loaded successfully")

    except Exception as e:
        log(f"ERROR loading ontology: {e}")
        log(f"Ontology must be present as '{ONTOLOGY_FILENAME}'")
        sys.exit(1)

    ensure_dir(ARTIFACTS_DIR)  # Create output directory

    log(f"Loading {DATA_FILENAME}...")
    data = None

    try:
        data = pd.read_csv(DATA_FILENAME)
        log(f"Data loaded successfully with {len(data)} rows and {len(data.columns)} columns")
    except Exception as e:
        log(f"Error loading {DATA_FILENAME} with default delimiter: {e}")
        try:
            data = pd.read_csv(DATA_FILENAME, sep=";")  # Try alternate delimiter
            log(f"Data loaded successfully with delimiter ';' - {len(data)} rows and {len(data.columns)} columns")
        except Exception as e:
            log(f"Error loading {DATA_FILENAME} with alternate delimiter: {e}")
            sys.exit(1)

    log("Extracting and saving unique values from each column...")

    column_stats = {}

    for column in data.columns:
        values = data[column].dropna().unique()
        column_stats[column] = {"unique_count": len(values), "null_count": data[column].isna().sum()}

        filename = column.lower().replace(" ", "_").replace("/", "_")
        save_path = os.path.join(ARTIFACTS_DIR, f"{filename}_values.txt")
        save_unique_values(values, save_path)

    stats_path = os.path.join(ARTIFACTS_DIR, "column_statistics.csv")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Column,Unique Values,Null Count\n")
        for column, stats in column_stats.items():
            f.write(f"{column},{stats['unique_count']},{stats['null_count']}\n")
    log(f"Saved column statistics to {stats_path}")

    log("Setting up classification mappings...")

    # Convert string mapping dictionaries to ontology objects
    fuel_type_map = {}
    for key, value in FUEL_TYPE_MAP.items():
        fuel_type_map[key] = getattr(onto, value)

    drive_type_map = {}
    for key, value in DRIVE_TYPE_MAP.items():
        drive_type_map[key] = getattr(onto, value)

    size_class_map = {}
    for key, value in SIZE_CLASS_MAP.items():
        size_class_map[key] = getattr(onto, value)

    body_style_map = {}
    for key, value in BODY_STYLE_MAP.items():
        body_style_map[key] = getattr(onto, value)

    boost_system_map = {}
    for key, value in BOOST_SYSTEM_MAP.items():
        boost_system_map[key] = getattr(onto, value)

    # For additional fuel types from dataset
    if "Fuel Type" in data.columns:
        fuel_values = data["Fuel Type"].dropna().unique()
        for fuel_type in fuel_values:
            if fuel_type and not pd.isna(fuel_type) and fuel_type not in fuel_type_map:
                fuel_id = fuel_type.replace(" ", "_").replace("-", "_").replace("/", "_")
                with onto:
                    try:
                        new_fuel_type = onto.FuelType(fuel_id)
                        fuel_type_map[fuel_type] = new_fuel_type
                        log(f"Added new FuelType: {fuel_type} -> {fuel_id}")
                    except Exception as e:
                        log(f"Error adding fuel type {fuel_id}: {e}")

    # For additional size classes from dataset
    if "Vehicle Size Class" in data.columns:
        size_values = data["Vehicle Size Class"].dropna().unique()
        for size_class in size_values:
            if size_class and not pd.isna(size_class) and size_class not in size_class_map:
                class_id = size_class.replace(" ", "_").replace("-", "_").replace(",", "").replace("/", "_")
                with onto:
                    try:
                        new_size_class = onto.VehicleSizeClass(class_id)
                        size_class_map[size_class] = new_size_class
                        log(f"Added new VehicleSizeClass: {size_class} -> {class_id}")
                    except Exception as e:
                        log(f"Error adding size class {class_id}: {e}")

    # For additional drive types from dataset
    if "Drive" in data.columns:
        drive_values = data["Drive"].dropna().unique()
        for drive_type in drive_values:
            if drive_type and not pd.isna(drive_type) and drive_type not in drive_type_map:
                drive_id = drive_type.replace(" ", "_").replace("-", "_").replace("/", "_")
                with onto:
                    try:
                        new_drive_type = onto.DriveType(drive_id)
                        drive_type_map[drive_type] = new_drive_type
                        log(f"Added new DriveType: {drive_type} -> {drive_id}")
                    except Exception as e:
                        log(f"Error adding drive type {drive_id}: {e}")

    # Create missing body style mappings if needed
    if "Vehicle Size Class" in data.columns:
        size_values = data["Vehicle Size Class"].dropna().unique()
        for size_class in size_values:
            if size_class and not pd.isna(size_class) and size_class not in body_style_map:
                body_style_map[size_class] = onto.Sedan
                log(f"Added default body style mapping for {size_class} -> Sedan")

    # Create Manufacturer individuals
    log("Creating manufacturer individuals...")
    manufacturer_map = {}
    if "Make" in data.columns:
        make_values = data["Make"].dropna().unique()
        for make in make_values:
            if make and not pd.isna(make):
                make_id = create_valid_id(make)
                with onto:
                    try:
                        manufacturer = onto.Manufacturer(make_id)
                        manufacturer_map[make] = manufacturer
                    except Exception as e:
                        log(f"Error adding manufacturer {make_id}: {e}")

                if len(manufacturer_map) % 20 == 0:
                    log(f"Created {len(manufacturer_map)} manufacturers...")
        log(f"Created all {len(manufacturer_map)} manufacturers")

    # Create Model Year individuals
    log("Creating model year individuals...")
    model_year_map = {}
    if "Year" in data.columns:
        year_values = data["Year"].dropna().unique()
        for year in year_values:
            if year and not pd.isna(year):
                year_id = f"Year_{int(year)}"
                with onto:
                    try:
                        year_individual = onto.ModelYear(year_id)
                        model_year_map[year] = year_individual
                    except Exception as e:
                        log(f"Error adding model year {year_id}: {e}")
        log(f"Created {len(model_year_map)} model years")

    # Now populate Vehicle instances - ONLY the base Vehicle class
    log("Populating Vehicle instances...")

    sample_size = min(MAX_VEHICLES, len(data))
    if sample_size < len(data):
        log(f"Processing {sample_size} sample vehicles out of {len(data)} total...")
        sample_data = data.sample(sample_size)
    else:
        log(f"Processing all {sample_size} vehicles...")
        sample_data = data

    processed_count = 0
    successful_count = 0

    # Prepare for categorizing vehicles - these will now be inferred
    vehicle_categories = {
        "PropulsionVehicle": 0,  # Top level
        "BodyStyleVehicle": 0,   # Top level
        "DriveTypeVehicle": 0,   # Top level
        
        # Propulsion-related
        "ConventionalFuelVehicle": 0,
        "AlternativeFuelVehicle": 0,
        "ElectrifiedVehicle": 0,
        "DieselVehicle": 0,
        "GasolineVehicle": 0,
        "RegularGasolineVehicle": 0,
        "PremiumGasolineVehicle": 0,
        "HybridElectricVehicle": 0,
        "PureElectricVehicle": 0,
        
        # Body style-related
        "PassengerVehicle": 0,
        "UtilityVehicle": 0,
        "SedanVehicle": 0,
        "CoupeVehicle": 0,
        "WagonVehicle": 0,
        "HatchbackVehicle": 0,
        "SUVVehicle": 0,
        "TruckVehicle": 0,
        "VanVehicle": 0,
        
        # Drive type-related
        "FrontWheelDriveVehicle": 0,
        "RearWheelDriveVehicle": 0,
        "AllWheelDriveVehicle": 0,
        "FourWheelDriveVehicle": 0,
        
        # Size-specific propulsion types
        "CompactDiesel": 0,
        "MidsizeDiesel": 0,
        "LargeDiesel": 0,
        "CompactRegular": 0,
        "MidsizeRegular": 0,
        "LargeRegular": 0,
        "CompactHybrid": 0,
        "MidsizeHybrid": 0,
        "LargeHybrid": 0,
        "CompactElectric": 0,
        "MidsizeElectric": 0,
        "LargeElectric": 0,
    }

    for i, row in sample_data.iterrows():
        if processed_count % BATCH_SIZE == 0:
            log(f"Processing vehicle batch {processed_count//BATCH_SIZE + 1}/{(sample_size+BATCH_SIZE-1)//BATCH_SIZE}...")

        row_id = row["ID"] if "ID" in row and not pd.isna(row["ID"]) else i
        id_with_zeros = f"{int(row_id):05d}"

        make = str(row["Make"]) if "Make" in row and not pd.isna(row["Make"]) else "Unknown"
        model = str(row["Model"]) if "Model" in row and not pd.isna(row["Model"]) else "Unknown"
        year = str(int(row["Year"])) if "Year" in row and not pd.isna(row["Year"]) else ""

        prefix = f"{make}_{model}_{year}".replace(" ", "_")
        prefix = create_valid_id(prefix)

        vehicle_id = f"{prefix}_{id_with_zeros}"

        try:
            with onto:
                # Create vehicle instance - ONLY use the base Vehicle class
                # The reasoner will automatically infer the subclasses
                vehicle = onto.Vehicle(vehicle_id)

                # Set data properties without using lists (for functional properties)
                if "Make" in row and not pd.isna(row["Make"]):
                    vehicle.make = str(row["Make"])

                if "Model" in row and not pd.isna(row["Model"]):
                    vehicle.model = str(row["Model"])

                if "Year" in row and not pd.isna(row["Year"]):
                    vehicle.year = int(row["Year"])

                if "Cylinders" in row and not pd.isna(row["Cylinders"]):
                    value = safe_numeric_conversion(row["Cylinders"])
                    if value is not None:
                        vehicle.cylinders = value

                if "You Save/Spend" in row and not pd.isna(row["You Save/Spend"]):
                    value = safe_numeric_conversion(row["You Save/Spend"])
                    if value is not None:
                        vehicle.savings = value

                if "Transmission" in row and not pd.isna(row["Transmission"]):
                    vehicle.transmission = str(row["Transmission"])

                if "Co2 Fuel Type1" in row and not pd.isna(row["Co2 Fuel Type1"]):
                    value = safe_numeric_conversion(row["Co2 Fuel Type1"])
                    if value is not None:
                        vehicle.co2Emissions = value

                if "Engine descriptor" in row and not pd.isna(row["Engine descriptor"]):
                    vehicle.engineDescriptor = str(row["Engine descriptor"])

                if "EPA Fuel Economy Score" in row and not pd.isna(row["EPA Fuel Economy Score"]):
                    value = safe_numeric_conversion(row["EPA Fuel Economy Score"])
                    if value is not None:
                        vehicle.epaFuelEconomyScore = value

                if "GHG Score" in row and not pd.isna(row["GHG Score"]):
                    value = safe_numeric_conversion(row["GHG Score"])
                    if value is not None:
                        vehicle.ghgScore = value

                if "Annual Petroleum Consumption For Fuel Type1" in row and not pd.isna(
                    row["Annual Petroleum Consumption For Fuel Type1"]
                ):
                    value = safe_numeric_conversion(row["Annual Petroleum Consumption For Fuel Type1"])
                    if value is not None:
                        vehicle.annualPetroleumConsumption = value

                if "City gasoline consumption" in row and not pd.isna(row["City gasoline consumption"]):
                    value = safe_numeric_conversion(row["City gasoline consumption"])
                    if value is not None:
                        vehicle.cityGasolineConsumption = value

                if "City electricity consumption" in row and not pd.isna(row["City electricity consumption"]):
                    value = safe_numeric_conversion(row["City electricity consumption"])
                    if value is not None:
                        vehicle.cityElectricityConsumption = value

                if "MPG Data" in row and not pd.isna(row["MPG Data"]):
                    vehicle.mpgData = str(row["MPG Data"])

                # Set electricity flag - CRITICAL for proper class inference
                has_electricity = False

                if "Fuel Type" in row and not pd.isna(row["Fuel Type"]):
                    fuel_type_str = str(row["Fuel Type"])
                    if fuel_type_str in ELECTRIC_FUEL_TYPES or "Electricity" in fuel_type_str:
                        has_electricity = True
                
                if "Electric motor" in row and not pd.isna(row["Electric motor"]):
                    has_electricity = True
                    vehicle.electricMotorSpec = str(row["Electric motor"])

                vehicle.hasElectricity = has_electricity

                # Set object properties without lists (for functional properties)
                # FuelType
                if "Fuel Type" in row and not pd.isna(row["Fuel Type"]):
                    if row["Fuel Type"] in fuel_type_map:
                        vehicle.hasFuelType = fuel_type_map[row["Fuel Type"]]

                # DriveType
                if "Drive" in row and not pd.isna(row["Drive"]):
                    if row["Drive"] in drive_type_map:
                        vehicle.hasDriveType = drive_type_map[row["Drive"]]

                # VehicleSizeClass
                if "Vehicle Size Class" in row and not pd.isna(row["Vehicle Size Class"]):
                    if row["Vehicle Size Class"] in size_class_map:
                        vehicle.hasSizeClass = size_class_map[row["Vehicle Size Class"]]

                # Manufacturer
                if "Make" in row and not pd.isna(row["Make"]):
                    if row["Make"] in manufacturer_map:
                        vehicle.hasManufacturer = manufacturer_map[row["Make"]]

                # ModelYear
                if "Year" in row and not pd.isna(row["Year"]):
                    if row["Year"] in model_year_map:
                        vehicle.hasModelYear = model_year_map[row["Year"]]

                # BodyStyle - derived from Vehicle Size Class
                if "Vehicle Size Class" in row and not pd.isna(row["Vehicle Size Class"]):
                    if row["Vehicle Size Class"] in body_style_map:
                        vehicle.hasBodyStyle = body_style_map[row["Vehicle Size Class"]]
                    else:
                        vehicle.hasBodyStyle = onto.Sedan  # Default to Sedan

                # BoostSystem
                if "T Charger" in row and not pd.isna(row["T Charger"]) and row["T Charger"] == "T":
                    vehicle.hasBoostSystem = onto.Turbocharger
                elif "S Charger" in row and not pd.isna(row["S Charger"]) and row["S Charger"] == "S":
                    vehicle.hasBoostSystem = onto.Supercharger
                else:
                    vehicle.hasBoostSystem = onto.NaturallyAspirated

                # MarketSegment - based on savings
                if "You Save/Spend" in row and not pd.isna(row["You Save/Spend"]):
                    market_segment = get_market_segment(row["You Save/Spend"])
                    vehicle.hasMarketSegment = getattr(onto, market_segment)
                else:
                    vehicle.hasMarketSegment = onto.StandardMarket  # Default market segment

            successful_count += 1

        except Exception as e:
            log(f"ERROR processing vehicle {vehicle_id}: {e}")

        processed_count += 1

    log(f"Vehicle population completed. Successfully processed {successful_count} out of {processed_count} vehicles.")

    log(f"Saving populated ontology to {OUTPUT_FILENAME}...")
    try:
        onto.save(file=OUTPUT_FILENAME, format="rdfxml")
        log(f"Ontology saved to {OUTPUT_FILENAME}")
    except Exception as e:
        log(f"Error saving populated ontology: {e}")

    # Run reasoner to classify vehicles - USING HERMIT INSTEAD OF PELLET
    log("Running HermiT reasoner to classify vehicles...")
    try:
        with onto:
            sync_reasoner_hermit(infer_property_values=True)
            
        log("Reasoning completed successfully")
    except Exception as e:
        log(f"Error during reasoning: {e}")

    # Calculate statistics after reasoning
    log("Ontology population statistics (after reasoning):")
    try:
        total_individuals = len(list(onto.individuals()))
        vehicle_instances = len(list(onto.Vehicle.instances()))
        log(f"Total number of individuals: {total_individuals}")
        log(f"Vehicle instances: {vehicle_instances}")

        # Count instances in each category after reasoning
        for category, _ in vehicle_categories.items():
            category_class = getattr(onto, category)
            instances_count = len(list(category_class.instances()))
            vehicle_categories[category] = instances_count
            if instances_count > 0:
                log(f"  {category}: {instances_count} instances")

    except Exception as e:
        log(f"Error calculating statistics: {e}")

    log("Ontology population process completed")


def main():
    """
    Main entry point for the ontology population script

    Executes the ontology population process and handles any
    top-level exceptions that might occur
    """
    try:
        populate_ontology()
    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()