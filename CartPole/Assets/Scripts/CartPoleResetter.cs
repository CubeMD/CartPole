using UnityEngine;

public class CartPoleResetter : MonoBehaviour
{
    private void OnTriggerEnter2D (Collider2D other)
    {
        if (other.transform.parent.TryGetComponent(out CartPoleAgent agent))
        {
            agent.EndEpisode();
            return;
        }
        
        if (other.transform.parent.TryGetComponent(out CartPoleVoluntaryAgent voluntaryAgent))
        {
            voluntaryAgent.EndEpisode();
        }
    }
}
