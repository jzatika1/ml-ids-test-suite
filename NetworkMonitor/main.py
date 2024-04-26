import asyncio
import logging

from network_monitor import NetworkMonitor

async def main():
    # Configuration
    models_directory = 'models'
    mappings_path = '../DataPreprocessing/model_mappings/mappings.json'

    # Initialize the NetworkMonitor with the directory and mappings path
    network_monitor = NetworkMonitor(models_directory=models_directory, mappings_path=mappings_path)
    
    # Start the network monitoring process
    try:
        await network_monitor.run()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Run the main function using asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program exited cleanly.")
