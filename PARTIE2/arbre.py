




class KDNode:
    def __init__(self, point=None, left=None, right=None):
        self.point = point  # Point dans l'espace k-dimensionnel
        self.left = left  # Sous-arbre gauche
        self.right = right  # Sous-arbre droit

def build_kdtree(points, depth=0):
    if not points:
        return None

    # Choix de la dimension (cyclique en fonction de la profondeur)
    k = len(points[0])  # Dimension de l'espace
    axis = depth % k

    # Tri des points selon la dimension actuelle et choix du point médian
    points.sort(key=lambda point: point[axis])
    median = len(points) // 2

    # Création du nœud et récursion
    return KDNode(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )
