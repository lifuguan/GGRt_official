import numpy as np

from hloc.utils.database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids

from ggrt.utils.union_find import UnionFind


def load_track_elements(database_path: str):
    db = COLMAPDatabase.connect(database_path)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM two_view_geometries")
    track_key_to_id = dict()
    track_elements, track_element_pairs = list(), list()

    for rows in cursor:
        pair_id, shape1, shape2, matches, config, F, E, H, qvec, tvec = rows
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        image_id1, image_id2 = image_id1 - 1, image_id2 - 1

        if matches == None:
            continue
        
        matches = blob_to_array(matches, np.uint32).reshape(shape1, shape2)
        for match_indices in matches:
            point2d_idx1, point2d_idx2 = match_indices[0], match_indices[1]
            track_element1 = TrackElement(image_id1, point2d_idx1)
            track_element2 = TrackElement(image_id2, point2d_idx2)
            
            track_key1 = str(image_id1) + '_' + str(point2d_idx1)
            track_key2 = str(image_id2) + '_' + str(point2d_idx2)
            
            if track_key1 in track_key_to_id.keys():
                track_id1 = track_key_to_id[track_key1]
            else:
                track_id1 = len(track_key_to_id)
                track_key_to_id[track_key1] = len(track_key_to_id) + 1
                track_elements.append(track_element1)
            
            if track_key2 in track_key_to_id.keys():
                track_id2 = track_key_to_id[track_key2]
            else:
                track_id2 = len(track_key_to_id)
                track_key_to_id[track_key2] = len(track_key_to_id) + 1
                track_elements.append(track_element2)

            track_element_pairs.append((track_id1, track_id2))
    
    return track_elements, track_element_pairs


def remove_key(d: dict, key):
    new_dict = dict(d)
    del new_dict[key]
    return new_dict


class TrackElement():
    def __init__(self, image_id: int, point2D_idx: int) -> None:
        """
        Args:
            image_id: The image in which the track element is observed.
            point2D_idx: The point in the image that the track element is observed.
        """
        self.image_id = image_id
        self.point2D_idx = point2D_idx


class TrackBuilder():
    def __init__(self, min_track_length=3, max_track_length=40) -> None:
        self.min_track_length = min_track_length
        self.max_track_length = max_track_length
        self.consistent_tracks = dict()

    def build(self, track_elements: list, track_element_pairs: list):
        print('Building tracks...')
        print(f'\tTotal track elements: {len(track_elements)}')
        print(f'\tTotal track element pairs: {len(track_element_pairs)}')
        num_track_elements = len(track_elements)
        
        # We use a truncated union find algorithm to build tracks due to outliers
        # exist in raw matches.
        finder = UnionFind(
            size=num_track_elements,
            max_num_per_set=self.max_track_length
        )

        # Union all connected tracks.
        for track_pair in track_element_pairs:
            finder.union(track_pair[0], track_pair[1])
        finder.validate()

        for track_id in range(num_track_elements):
            root_id = finder.find_root(track_id)
            if root_id not in self.consistent_tracks.keys():
                self.consistent_tracks[root_id] = []
            self.consistent_tracks[root_id].append(track_elements[track_id])
        
        print(f'\tTotal tracks: {len(self.consistent_tracks)}')
        print(f'\tmean track length: {self.mean_track_length}')

    def filter(self):
        # Filter inconsistent tracks and short tracks.
        num_small_tracks = 0
        num_inconsistent_track_elements = 0
        print('Filtering tracks...')

        track_root_ids = self.consistent_tracks.copy().keys()

        for root_id in track_root_ids:
            # If track.length  < min_track_length or track.length > max_track_length,
            # we should discard this track.
            if len(self.consistent_tracks[root_id]) < self.min_track_length:
                self.consistent_tracks.pop(root_id)
                num_small_tracks += 1
                continue
            
            candidate_tracks = self.consistent_tracks[root_id]
            consistent_tracks = list()
            num_track_elements = len(candidate_tracks)
            image_ids = dict()
            for i in range(num_track_elements):
                track_element = candidate_tracks[i]
                # Do not add the track element if the track already contains a track
                # element from the same image.
                if track_element.image_id in image_ids:
                    num_inconsistent_track_elements += 1
                    continue
                
                image_ids[track_element.image_id] = 1
                consistent_tracks.append(track_element)
            
            if len(candidate_tracks) != len(consistent_tracks):
                self.consistent_tracks.pop(root_id)
                # In case of track becomes short after removing ambiguous tracks.
                if len(consistent_tracks) >= self.min_track_length:
                    self.consistent_tracks[root_id] = consistent_tracks
        
        print(f'\t{num_small_tracks} small tracks are removed.')
        print(f'\t{num_inconsistent_track_elements} inconsistent track elements are removed.')
        print(f'\t{len(self.consistent_tracks)} consistent tracks are reserved.')
        print(f'\tmean track length: {self.mean_track_length}')

    @property
    def mean_track_length(self):
        mean_track_length = 0.
        for key in self.consistent_tracks.keys():
            mean_track_length += len(self.consistent_tracks[key])
        mean_track_length /= len(self.consistent_tracks)
        return round(mean_track_length, 2)

    def write_to_file(self, filename: str):
        with open(filename, 'w') as f:
            for key in self.consistent_tracks.keys():
                f.write(f'{key} {int(len(self.consistent_tracks[key]))}\n')
                for track in self.consistent_tracks[key]:
                    f.write(f'{int(track.image_id)} {int(track.point2D_idx)} ')
                f.write('\n')
        f.close()

    def read_from_file(self, filename: str, image_ids=None):
        with open(filename, 'r') as f:
            line = f.readline().rstrip('\n').split(' ')
            while line and len(line) > 1:
                track_id, num_track_elements = int(line[0]), int(line[1])
                self.consistent_tracks[track_id] = list()
                line = f.readline().rstrip('\n').split(' ')
                for i in range(num_track_elements):
                    image_id, point2D_idx = int(line[2 * i + 0]), int(line[2 * i + 1])

                    # Only loading tracks for the specified images.
                    if image_ids is not None and image_id not in image_ids:
                        continue

                    track_element = TrackElement(image_id, point2D_idx)
                    self.consistent_tracks[track_id].append(track_element)
                line = f.readline().rstrip('\n').split(' ')
        f.close()
