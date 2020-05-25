import logging
import numpy as np

from abstract.device import Device


class TouchScreenDevice(Device):
    def __init__(self, layout_config, ignore_key='-'):
        super(TouchScreenDevice, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.load_layout(layout_config)
        unique_list = np.unique(self.layout)
        self.keys = np.delete(unique_list, np.where(unique_list == ignore_key))
        self.sensor_loc = None

    def get_coordinate(self, char):
        """
        Returns the (row, column) index corresponding to the character.
        :param char: character for which the coordinate is asked.
        :return: list [row, column].
        """
        coord = np.where(self.layout == char)
        return np.hstack(coord)

    def get_character(self, row, column):
        """
        Returns the string character corresponding to the (row, column).
        :param row: int value of the row.
        :param column: int value of the column.
        :return: string character.
        """
        if 0 <= row < self.layout.shape[0] and 0 <= column < self.layout.shape[1]:
            return self.layout[row][column]
        else:
            self.logger.error('row {%d} or column {%d} is out of bound' % (row, column))

    def get_random_key(self):
        """
        Returns a random string character present in the layout.
        :return: string character.
        """
        return np.random.choice(self.keys, 1, replace=True)

    def initialise_sensor_position(self):
        """
        Initialise the sensor position on the screen.
        :return:
        """
        random_row = np.random.randint(0, self.layout.shape[0])
        random_column = np.random.randint(0, self.layout.shape[1])
        self.logger.debug('setting the sensor to row {%d} and column {%d}' % (random_row, random_column))
        self.sensor_loc = [random_row, random_column]
        return self.sensor_loc

    def start(self):
        """
        Function to initialise the device.

        :param sample: boolean flag for device to generate a random target key or not.
        :return: string char or None
        """
        self.logger.debug('Starting the touch screen device.')
        # Initialise the sensor position to a random location.
        sensor_loc = self.initialise_sensor_position()

        return sensor_loc


